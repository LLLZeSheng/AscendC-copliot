// [[[ REPLACE_START: GluSingleTilingCalculator::CalcTiling ]]]
bool GluSingleTilingCalculator::CalcTiling(uint32_t totalCore, uint64_t ubSize, int32_t dtype,  platform_ascendc::SocVersion socVersion_)
{
    totalAvailableCore = totalCore;
    if (!GetLengthByType(dtype, inputDTypeLen)) {
        OP_LOGI(opName_, "CalcTiling Unsupported input data type %d", dtype);
        return false;
    }
    ubMinBlockLen = UB_MIN_BLOCK_SIZE / inputDTypeLen; // min block size
    cacheLineLen = L2_CACHE_LINE_SIZE / inputDTypeLen; // bandwidth max efficiency
    alignPackLen = cacheLineLen; // 默认512对齐，策略可调整
    OP_LOGI(opName_, "CalcTiling GetLengthByType:%u ubMinBlockLen:%u cacheLineLen:%u alignPackLen:%u", inputDTypeLen, ubMinBlockLen, cacheLineLen, alignPackLen);
    // Is 32-byte aligned for split colLen?
    tilingData->set_is32BAligned(tilingData->get_colLen() % ubMinBlockLen == 0);
    // 310p not support Non-64B
    uint32_t blockSizeOf64B = BLOCK_SIZE_OF_64B / inputDTypeLen;
    if (((socVersion_ == platform_ascendc::SocVersion::ASCEND310P)) && (tilingData->get_colLen() % blockSizeOf64B != 0)) {
        OP_LOGE(opName_, "input shape is not support Non-64B aligned");
        return false;
    }
    // 先计算开启double buffer的tiling参数
    tilingData->set_isDoubleBuffer(1);
    GluSingleTilingOptParam optTilingDb;
    // 判断buffer = 2时是否计算成功
    if (!CalcOptTiling<Glu_Flag, 2>(ubSize, dtype, optTilingDb)) {
        return false;
    }
    GluSingleTilingOptParam *optTiling = &optTilingDb;
    // 如果double buffer开启的tiling参数中，每个核需要处理的tileNum等于2，尝试关闭double buffer;
    // 若关闭double buffer后只需要搬运1次数据，且使用的核没有减少, 则使用关闭double buffer的tiling
    // 判断tileNumPerCoer是否为2
    if (optTilingDb.tileNumPerCore == static_cast<uint64_t>(2)) {
        GluSingleTilingOptParam optTilingNoDb;
        if (CalcOptTiling<Glu_Flag, 1>(ubSize, dtype, optTilingNoDb) &&
            (optTilingNoDb.tileNumPerCore == static_cast<uint64_t>(1)) && (optTilingNoDb.totalUsedCoreNum >= optTilingDb.totalUsedCoreNum)) {
            optTiling = &optTilingNoDb;
            tilingData->set_isDoubleBuffer(0);
        }
    }
    // 记录最优的结果
    tilingData->set_baseRowLen(optTiling->optBaseRowLen);
    tilingData->set_baseColLen(optTiling->optBaseColLen);
    totalUsedCoreNum_ = optTiling->totalUsedCoreNum;
    OP_LOGI(opName_, "CalcTilingRES baseRowLen:%u baseColLen:%u", optTiling->optBaseRowLen, optTiling->optBaseColLen);
    return true;
}
// [[[ REPLACE_END ]]]


// [[[ REPLACE_START: GluSingleTilingCalculator::CalcOptTiling ]]]
bool GluSingleTilingCalculator::CalcOptTiling(uint64_t ubSize, int32_t dtype, GluSingleTilingOptParam& optTiling) const
{
    // 计算maxTilingLen
    if (!CalcUbMaxTileLen<Glu_Flag, bufferNum>(ubSize, dtype, optTiling)) {
        return false;
    }
    // 计算最优的base块形状
    if (!CalcOptBaseShape<Glu_Flag>(optTiling)) {
        return false;
    }
    return true;
}
// [[[ REPLACE_END ]]]


// [[[ REPLACE_START: GluSingleTilingCalculator::CalcOptBaseShape ]]]
inline bool GluSingleTilingCalculator::CalcOptBaseShape(GluSingleTilingOptParam& optTiling) const
{
    uint32_t baseColLen_ = getBaseColLenUpBound(optTiling);
    OP_LOGI(opName_, "CalcOptBaseShape init baseColLen : %u", baseColLen_);
    if (MustBeSingleBaseRowLen(baseColLen_)) {
        SaveOptBaseShape(static_cast<uint32_t>(1), baseColLen_, optTiling);
        return true;
    }

    while(true) {
        // colLen非32B对齐时，数据copy到ub时，每一行的尾部会补齐32B
        uint32_t baseRowlen_ = std::min(optTiling.maxTileLen / AlignUp<uint32_t>(baseColLen_, ubMinBlockLen), getBaseRowLenUpBound());
        if (isInvalidBaseShape(baseRowlen_, baseColLen_)) {
            OP_LOGI(opName_, "CalcOptBaseShape baseRowln:%u or baseColLen:%u is invalid. optTotalTileNum:%lu end",
                    baseRowlen_, baseColLen_, optTiling.optTotalTileNum);
            // optTotalTileNum有效，则前面有最优解，返回true;否则返回false
            return (optTiling.optTotalTileNum > static_cast<uint64_t>(0));
        }
        // 保存较优的base shape
        if (isValidTailCol(baseRowlen_, baseColLen_)) {
            SaveOptBaseShape(baseRowlen_, baseColLen_, optTiling);
        }

        // baseColLen已经到达下限 或者 baseRowlen已经达到上限，无法继续调整，结束
        if (baseColLen_ <= alignPackLen || (baseRowlen_ >= getBaseRowLenUpBound())) {
            return true; // baseColLen无法继续调整了，结束
        }
        // 继续调整baseColLen
        // baseColLen为若alignPackLen的整数倍，则baseColLen减少1个alignPackLen的长度
        // 否则baseColLen减少到alignPackLen的整数倍（最接近的）
        if (baseColLen_ % alignPackLen == static_cast<uint32_t>(0)) {
            baseColLen_ -= alignPackLen;
        } else {
            baseColLen_ = AlignDown<uint32_t>(baseColLen_, alignPackLen);
        }
    }
}
// [[[ REPLACE_END ]]]


// [[[ REPLACE_START: GluSingleTilingCalculator::SaveOptBaseShape ]]]
inline void GluSingleTilingCalculator::SaveOptBaseShape(uint32_t baseRowLen_, uint32_t baseColLen_, GluSingleTilingOptParam& optTiling) const
{
    uint64_t totalTileNum = DivCeil<uint64_t>(tilingData->get_rowLen(), (baseRowLen_)) * DivCeil<uint64_t>(tilingData->get_colLen(), (baseColLen_));
    uint64_t baseSize = static_cast<uint64_t>(baseRowLen_) * baseColLen_;
    if (baseRowLen_ == static_cast<uint32_t>(0) || baseColLen_ == static_cast<uint32_t>(0)) {
        OP_LOGE(opName_, "SaveOptBaseShape devide by 0 baseRowLen_:%u baseColLen_:%u", baseRowLen_, baseColLen_);
        return;
    }
    uint64_t baseTileNum = (tilingData->get_rowLen() / baseRowLen_) * (tilingData->get_colLen() / baseColLen_);
    uint32_t totalUsedCoreNum = std::min(totalTileNum, static_cast<uint64_t>(totalAvailableCore));
    if ((optTiling.optTotalTileNum == static_cast<uint64_t>(0)) ||
        (totalUsedCoreNum > optTiling.totalUsedCoreNum) ||
        ((totalUsedCoreNum == optTiling.totalUsedCoreNum) && (totalTileNum < optTiling.optTotalTileNum)) ||
        ((totalUsedCoreNum == optTiling.totalUsedCoreNum) && (totalTileNum == optTiling.optTotalTileNum) && (baseSize > optTiling.optBaseSize)) ||
        ((totalUsedCoreNum == optTiling.totalUsedCoreNum) && (totalTileNum == optTiling.optTotalTileNum) && (baseSize == optTiling.optBaseSize) && (baseTileNum > optTiling.optBaseTileNum))) {
        optTiling.optBaseRowLen = baseRowLen_;
        optTiling.optBaseColLen = baseColLen_;
        optTiling.optTotalTileNum = totalTileNum;
        optTiling.optBaseSize = baseSize;
        optTiling.optBaseTileNum = baseTileNum;
        optTiling.totalUsedCoreNum = totalUsedCoreNum;
        optTiling.tileNumPerCore = DivCeil<uint64_t>(totalTileNum, totalUsedCoreNum);;
    }
}
// [[[ REPLACE_END ]]]

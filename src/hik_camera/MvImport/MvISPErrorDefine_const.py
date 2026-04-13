#!/usr/bin/env python
# -*- coding: utf-8 -*-


## @~chinese 通用类型
MV_ALG_OK                    = 0x00000000  ## @~chinese处理正确
MV_ALG_ERR                   = 0x10000000  ## @~chinese不确定类型错误

## @~chinese 能力检查
MV_ALG_E_ABILITY_ARG         = 0x10000001  ## @~chinese能力集中存在无效参数

## @~chinese 内存检查
MV_ALG_E_MEM_NULL            = 0x10000002  ## @~chinese内存地址为空
MV_ALG_E_MEM_ALIGN           = 0x10000003  ## @~chinese内存对齐不满足要求
MV_ALG_E_MEM_LACK            = 0x10000004  ## @~chinese内存空间大小不够
MV_ALG_E_MEM_SIZE_ALIGN      = 0x10000005  ## @~chinese内存空间大小不满足对齐要求
MV_ALG_E_MEM_ADDR_ALIGN      = 0x10000006  ## @~chinese内存地址不满足对齐要求

## @~chinese 图像检查
MV_ALG_E_IMG_FORMAT          = 0x10000007  ## @~chinese图像格式不正确或者不支持
MV_ALG_E_IMG_SIZE            = 0x10000008  ## @~chinese图像宽高不正确或者超出范围
MV_ALG_E_IMG_STEP            = 0x10000009  ## @~chinese图像宽高与step参数不匹配
MV_ALG_E_IMG_DATA_NULL       = 0x1000000A  ## @~chinese图像数据存储地址为空

## @~chinese 输入输出参数检查
MV_ALG_E_CFG_TYPE            = 0x1000000B  ## @~chinese设置或者获取参数类型不正确
MV_ALG_E_CFG_SIZE            = 0x1000000C  ## @~chinese设置或者获取参数的输入、输出结构体大小不正确
MV_ALG_E_PRC_TYPE            = 0x1000000D  ## @~chinese处理类型不正确
MV_ALG_E_PRC_SIZE            = 0x1000000E  ## @~chinese处理时输入、输出参数大小不正确
MV_ALG_E_FUNC_TYPE           = 0x1000000F  ## @~chinese子处理类型不正确
MV_ALG_E_FUNC_SIZE           = 0x10000010  ## @~chinese子处理时输入、输出参数大小不正确

## @~chinese 运行参数检查
MV_ALG_E_PARAM_INDEX         = 0x10000011  ## @~chineseindex参数不正确
MV_ALG_E_PARAM_VALUE         = 0x10000012  ## @~chinesevalue参数不正确或者超出范围
MV_ALG_E_PARAM_NUM           = 0x10000013  ## @~chineseparam_num参数不正确

## @~chinese 接口调用检查
MV_ALG_E_NULL_PTR            = 0x10000014  ## @~chinese函数参数指针为空
MV_ALG_E_OVER_MAX_MEM        = 0x10000015  ## @~chinese超过限定的最大内存
MV_ALG_E_CALL_BACK           = 0x10000016  ## @~chinese回调函数出错

## @~chinese 算法库加密相关检查
MV_ALG_E_ENCRYPT             = 0x10000017  ## @~chinese加密错误
MV_ALG_E_EXPIRE              = 0x10000018  ## @~chinese算法库使用期限错误

## @~chinese 内部模块返回的基本错误类型
MV_ALG_E_BAD_ARG             = 0x10000019  ## @~chinese参数范围不正确
MV_ALG_E_DATA_SIZE           = 0x1000001A  ## @~chinese数据大小不正确
MV_ALG_E_STEP                = 0x1000001B  ## @~chinese数据step不正确

## @~chinese cpu指令集支持错误码
MV_ALG_E_CPUID               = 0x1000001C  ## @~chinesecpu不支持优化代码中的指令集

MV_ALG_WARNING               = 0x1000001D  ## @~chinese警告

MV_ALG_E_TIME_OUT            = 0x1000001E  ## @~chinese算法库超时
MV_ALG_E_LIB_VERSION         = 0x1000001F  ## @~chinese算法版本号出错
MV_ALG_E_MODEL_VERSION       = 0x10000020  ## @~chinese模型版本号出错
MV_ALG_E_GPU_MEM_ALLOC       = 0x10000021  ## @~chineseGPU内存分配错误
MV_ALG_E_FILE_NON_EXIST      = 0x10000022  ## @~chinese文件不存在
MV_ALG_E_NONE_STRING         = 0x10000023  ## @~chinese字符串为空
MV_ALG_E_IMAGE_CODEC         = 0x10000024  ## @~chinese图像解码器错误
MV_ALG_E_FILE_OPEN           = 0x10000025  ## @~chinese打开文件错误
MV_ALG_E_FILE_READ           = 0x10000026  ## @~chinese文件读取错误
MV_ALG_E_FILE_WRITE          = 0x10000027  ## @~chinese文件写错误
MV_ALG_E_FILE_READ_SIZE      = 0x10000028  ## @~chinese文件读取大小错误
MV_ALG_E_FILE_TYPE           = 0x10000029  ## @~chinese文件类型错误
MV_ALG_E_MODEL_TYPE          = 0x1000002A  ## @~chinese模型类型错误
MV_ALG_E_MALLOC_MEM          = 0x1000002B  ## @~chinese分配内存错误
MV_ALG_E_BIND_CORE_FAILED    = 0x1000002C  ## @~chinese线程绑核失败

## @~chinese 降噪特有错误码
MV_ALG_E_DENOISE_NE_IMG_FORMAT        = 0x10402001  ## @~chinese噪声特性图像格式错误
MV_ALG_E_DENOISE_NE_FEATURE_TYPE      = 0x10402002  ## @~chinese噪声特性类型错误
MV_ALG_E_DENOISE_NE_PROFILE_NUM       = 0x10402003  ## @~chinese噪声特性个数错误
MV_ALG_E_DENOISE_NE_GAIN_NUM          = 0x10402004  ## @~chinese噪声特性增益个数错误
MV_ALG_E_DENOISE_NE_GAIN_VAL          = 0x10402005  ## @~chinese噪声曲线增益值输入错误
MV_ALG_E_DENOISE_NE_BIN_NUM           = 0x10402006  ## @~chinese噪声曲线柱数错误
MV_ALG_E_DENOISE_NE_INIT_GAIN         = 0x10402007  ## @~chinese噪声估计初始化增益设置错误
MV_ALG_E_DENOISE_NE_NOT_INIT          = 0x10402008  ## @~chinese噪声估计未初始化
MV_ALG_E_DENOISE_COLOR_MODE           = 0x10402009  ## @~chinese颜色空间模式错误
MV_ALG_E_DENOISE_ROI_NUM              = 0x1040200a  ## @~chinese图像ROI个数错误
MV_ALG_E_DENOISE_ROI_ORI_PT           = 0x1040200b  ## @~chinese图像ROI原点错误
MV_ALG_E_DENOISE_ROI_SIZE             = 0x1040200c  ## @~chinese图像ROI大小错误
MV_ALG_E_DENOISE_GAIN_NOT_EXIST       = 0x1040200d  ## @~chinese输入的相机增益不存在(增益个数已达上限)
MV_ALG_E_DENOISE_GAIN_BEYOND_RANGE    = 0x1040200e  ## @~chinese输入的相机增益不在范围内
MV_ALG_E_DENOISE_NP_BUF_SIZE          = 0x1040200f  ## @~chinese输入的噪声特性内存大小错误

## @~chinese 去紫边特有错误码
MV_ALG_E_PFC_ROI_PT                  = 0x10405000  ## @~chinese去紫边算法ROI原点错误
MV_ALG_E_PFC_ROI_SIZE                = 0x10405001  ## @~chinese去紫边算法ROI大小错误
MV_ALG_E_PFC_KERNEL_SIZE             = 0x10405002  ## @~chinese去紫边算法滤波核尺寸错误


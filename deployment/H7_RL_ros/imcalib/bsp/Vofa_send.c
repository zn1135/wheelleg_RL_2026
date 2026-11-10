/*
 * 文件职责：板载USB完整采集与调试UART八通道VOFA分别打包发送。
 * 调用链：后台服务 → Vofa_Debug_Send → USB CDC / UART TX DMA。
 * 同一次USB提交：原n个float/帧尾前缀，再追加H7OUT001输出批次和H7LAT001时序批次。
 * 静态缓冲在异步发送完成前不能覆盖；USB忙不出队，队满单独报告，不接收控制命令。
 */
#include "Vofa_send.h"
#include "usbd_cdc_if.h"
#include "output_capture.h"
#include "latency_capture.h"
#include "dma_cache.h"
#include <string.h>

static const uint8_t tail[4] = {0x00, 0x00, 0x80, 0x7F};

/* UART只发四关节位置/反馈力矩；DMA完成前保留上一帧，不占用USB缓冲或采集队列。 */
uint8_t Vofa_Uart_Send(const float *data)
{
    static uint8_t buf[8u * sizeof(float) + 4u] __attribute__((aligned(DMA_CACHE_LINE_SIZE)));

    if (VOFA_UART->gState != HAL_UART_STATE_READY)
    {
        return 0u;
    }
    memcpy(buf, data, 8u * sizeof(float));
    memcpy(buf + 8u * sizeof(float), tail, sizeof(tail));
    Dma_Cache_Clean_Tx(buf, sizeof(buf));
    return HAL_UART_Transmit_DMA(VOFA_UART, buf, sizeof(buf)) == HAL_OK;
}

/* 校验输入 → 检查USB已枚举且空闲 → 打包 → 提交；失败/忙不等待。 */
uint8_t Vofa_Send(const float *data, uint8_t n)
{
    static uint8_t buf[VOFA_MAX_CH * sizeof(float) + 4 + OUTPUT_CAPTURE_PACKET_MAX + LATENCY_PACKET_MAX];
    uint16_t data_len;

    if (n == 0u || n > VOFA_MAX_CH)
    {
        return 0u;
    }
    /* 必须在memcpy前检查，USB完成回调前仍持有上一帧缓冲。 */
    if (!CDC_Transmit_Ready_HS())
    {
        return 0u;
    }
    data_len = (uint16_t)(n * sizeof(float));
    memcpy(buf, data, data_len);
    memcpy(buf + data_len, tail, sizeof(tail));
    data_len = (uint16_t)(data_len + sizeof(tail));
    data_len = (uint16_t)(data_len + Output_Capture_Packet(buf + data_len));
    data_len = (uint16_t)(data_len + Latency_Capture_Packet(buf + data_len));
    if (CDC_Transmit_HS(buf, data_len) != USBD_OK)
    {
        return 0u;
    }
    Output_Capture_Commit();
    Latency_Capture_Commit();
    return 1u;
}

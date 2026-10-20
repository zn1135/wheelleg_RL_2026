/*
 * Jetson -> H7探测包："J7P1" + uint32序号 + uint32随机数 + uint16 CRC，均小端。
 * CRC16/CCITT-FALSE校验前12字节。解析只更新通信状态，不接触模型、动作或电机。
 * 主机最多10Hz、一次只等一个响应；comm每拍读取DMA新字节，跨环尾和分包保持解析状态。
 * DMA环形区256字节，轮询间不得积压一整圈；本接口不用于连续高速数据流。
 */
#include "jetson_uart.h"
#include "Vofa_send.h"
#include "dma_cache.h"
#include <string.h>

#define JETSON_UART_DMA_BYTES 256u

jetson_uart_status_t jetson_uart_status;
static uint8_t rx_dma[JETSON_UART_DMA_BYTES] __attribute__((aligned(DMA_CACHE_LINE_SIZE)));
static uint8_t packet[JETSON_UART_PACKET_BYTES];
static uint16_t read_pos;
static uint8_t packet_size;

static uint16_t Jetson_Crc16(const uint8_t *data, uint16_t size)
{
    uint16_t crc = 0xFFFFu;
    for (uint16_t i = 0u; i < size; i++)
    {
        crc ^= (uint16_t)data[i] << 8;
        for (uint8_t bit = 0u; bit < 8u; bit++)
        {
            crc = (crc & 0x8000u) ? (uint16_t)((crc << 1) ^ 0x1021u) : (uint16_t)(crc << 1);
        }
    }
    return crc;
}

/* 按字节解码，避免接收缓冲未对齐时直接读取uint32。 */
static uint32_t Jetson_U32(const uint8_t *data)
{
    return (uint32_t)data[0] | ((uint32_t)data[1] << 8) | ((uint32_t)data[2] << 16) | ((uint32_t)data[3] << 24);
}

/* 噪声/坏CRC后逐字节重新找包头；不把一次DMA读取当成完整协议包。 */
static void Jetson_Receive_Byte(uint8_t value)
{
    uint16_t expected;

    packet[packet_size++] = value;
    while (packet_size >= 4u && memcmp(packet, "J7P1", 4u) != 0)
    {
        memmove(packet, packet + 1u, --packet_size);
    }
    if (packet_size != JETSON_UART_PACKET_BYTES)
    {
        return;
    }
    expected = (uint16_t)packet[12] | ((uint16_t)packet[13] << 8);
    if (Jetson_Crc16(packet, 12u) == expected)
    {
        jetson_uart_status.sequence = Jetson_U32(packet + 4u);
        jetson_uart_status.token = Jetson_U32(packet + 8u);
        jetson_uart_status.received++;
        packet_size = 0u;
    }
    else
    {
        jetson_uart_status.crc_errors++;
        memmove(packet, packet + 1u, --packet_size);
    }
}

void Jetson_Uart_Init(void)
{
    read_pos = 0u;
    packet_size = 0u;
    Dma_Cache_Invalidate_Rx(rx_dma, sizeof(rx_dma));
    if (HAL_UART_Receive_DMA(VOFA_UART, rx_dma, sizeof(rx_dma)) == HAL_OK)
    {
        /* comm轮询消费；保留TC中断，HAL接收错误后的异步DMA终止依赖它完成。 */
        __HAL_DMA_DISABLE_IT(VOFA_UART->hdmarx, DMA_IT_HT);
    }
}

void Jetson_Uart_Process(void)
{
    uint16_t write_pos;

    if (VOFA_UART->RxState != HAL_UART_STATE_BUSY_RX)
    {
        /* HAL遇到接收错误会终止DMA；恢复时清半包，不回报旧请求作为新包。 */
        jetson_uart_status.rx_restarts++;
        Jetson_Uart_Init();
        return;
    }
    write_pos = (uint16_t)(JETSON_UART_DMA_BYTES - __HAL_DMA_GET_COUNTER(VOFA_UART->hdmarx));
    write_pos %= JETSON_UART_DMA_BYTES;
    if (read_pos == write_pos)
    {
        return;
    }
    Dma_Cache_Invalidate_Rx(rx_dma, sizeof(rx_dma));
    while (read_pos != write_pos)
    {
        Jetson_Receive_Byte(rx_dma[read_pos]);
        read_pos = (uint16_t)((read_pos + 1u) % JETSON_UART_DMA_BYTES);
    }
}

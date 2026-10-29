/*
 * Jetson串口连通性协议：只接收带CRC的探测包，不接收动作或改变控制许可。
 * comm任务更新状态，后台在短临界区读取并通过镜像预留槽回报。
 */
#ifndef JETSON_UART_H
#define JETSON_UART_H

#include <stdint.h>

#define JETSON_UART_PROTOCOL 1u
#define JETSON_UART_PACKET_BYTES 14u

typedef struct {
    uint32_t sequence;       /* 最近合法探测包的序号。 */
    uint32_t token;          /* 原样回报主机随机数，避免把旧回报当作本次响应。 */
    uint32_t received;       /* CRC通过的包数，重复探测也计数。 */
    uint32_t crc_errors;
    uint32_t rx_restarts;    /* DMA异常后的重新启动次数；首次初始化不计。 */
} jetson_uart_status_t;

extern jetson_uart_status_t jetson_uart_status;

/* main初始化，随后由comm任务轮询DMA环形区；无需IDLE中断或额外任务。 */
void Jetson_Uart_Init(void);
void Jetson_Uart_Process(void);

#endif

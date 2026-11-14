/* UART10专用RL链路；通信任务接收，策略节拍发布，执行拍回报首次PD。 */
#ifndef JETSON_UART_H
#define JETSON_UART_H
#include <stdint.h>
void Jetson_Uart_Init(void);
void Jetson_Uart_Process(void);
void Jetson_Uart_Publish(void);
void Jetson_Uart_Action_Clear(uint32_t reset);
void Jetson_Uart_Pd(uint32_t sequence, uint32_t reset);
uint8_t Jetson_Uart_Blocked(void);
#endif

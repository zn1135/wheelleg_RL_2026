/*
 * 文件职责：板载USB完整采集及独立UART八通道VOFA发送接口。
 * 输入：镜像浮点数组及通道数量；同一次USB提交追加完整输出缓存批次。
 * USB未枚举或忙时不出队，完成前缓冲由实现文件持有；不提供控制接收。
 * VOFA_UART用于八通道TX DMA和既有探测接收；两路发送缓冲相互独立。
 */
#ifndef VOFA_SEND_H
#define VOFA_SEND_H

#include "main.h"
#include "usart.h"
#include "machine_config.h"

#define VOFA_MAX_CH 160u /* 镜像v2固定160路；旧48路布局仅作历史参考。 */

/* 调试UART：大机USART10接烧录器串口，小机沿用USART1；USB镜像不使用此宏。 */
#if MACHINE_VOFA_PORT == 7u
#define VOFA_UART (&huart7)
#elif MACHINE_VOFA_PORT == 8u
#define VOFA_UART (&huart8)
#elif MACHINE_VOFA_PORT == 9u
#define VOFA_UART (&huart9)
#elif MACHINE_VOFA_PORT == 10u
#define VOFA_UART (&huart10)
#elif MACHINE_VOFA_PORT == 1u
#define VOFA_UART (&huart1)
#else
#error "MACHINE_VOFA_PORT must be 7 (UART7), 8 (UART8), 9 (UART9), 10 (USART10) or 1 (USART1)"
#endif

/* 成功返回1表示USB提交成功，不代表Jetson已收到；波特率不决定USB线速。 */
uint8_t Vofa_Send(const float *data, uint8_t n);

/* 固定八通道JustFloat，位置rad/反馈力矩N·m相邻；成功仅表示DMA已接受，不是主机收妥。 */
uint8_t Vofa_Uart_Send(const float *data);

#endif

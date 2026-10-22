/*
 * 文件职责：板载USB CDC的VOFA JustFloat发送接口。
 * 输入：镜像浮点数组及通道数量；同一次USB提交追加完整输出缓存批次。
 * USB未枚举或忙时不出队，完成前缓冲由实现文件持有；不提供控制接收。
 * 下方VOFA_UART仅保留给既有UART探测接收器，不再决定镜像发送路径。
 */
#ifndef VOFA_SEND_H
#define VOFA_SEND_H

#include "main.h"
#include "usart.h"
#include "machine_config.h"

#define VOFA_MAX_CH 160u /* 镜像v2固定160路；旧48路布局仅作历史参考。 */

/* 既有UART探测接收器的端口；USB镜像不使用此宏。 */
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

#endif

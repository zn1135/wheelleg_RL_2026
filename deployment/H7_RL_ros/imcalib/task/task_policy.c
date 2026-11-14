/* 100Hz仅发布最新状态；观测历史与网络推理由Jetson C++维护。 */
#include "robot_control.h"
#include "robot_state.h"
#include "jetson_uart.h"
void ctrl_task_init(void) { rl_control.policy.ready=0u; }
void ctrl_task_body(void) { if(machine->rl.configured) Jetson_Uart_Publish(); }

/*
 * 文件职责：FDCAN公共注册、接收分发与发送服务（can_bus.c）。
 * 输入/来源：CubeMX FDCAN句柄、注册ID及驱动收发请求。
 * 输出/去向：HAL中断读取帧并派发协议回调；统一构造发送帧、统计接收和错误。
 * 关键边界：本层不解释DM/DJI物理协议或策略意图；HAL_OK仅表明发送入队调用成功。
 * 阅读入口：md/file-guide.md；参数当前值以源码为准。
 */

#include "can_bus.h"
#include "FreeRTOS.h"
#include "task.h"
#include "latency_capture.h"
#include "mono_ns.h"
#include <string.h>

static can_bus_t can_bus[CAN_BUS_NUM];
/* 每个硬件TX buffer独立保存原提交信息；不能用最新全局命令配完成IRQ。 */
static latency_record_t latency_tx[CAN_BUS_NUM][32];
static uint8_t latency_tx_active[CAN_BUS_NUM][32];
static uint32_t latency_tx_sequence;

static void Can_Latency_Finish(uint32_t bus, uint32_t slot, uint64_t now_ns, uint8_t result)
{
    latency_record_t record;
    if (!latency_tx_active[bus][slot]) { return; }
    record = latency_tx[bus][slot];
    record.end_ns = now_ns;
    record.flags = (uint8_t)((record.flags & 0x80u) | result);
    latency_tx_active[bus][slot] = 0u;
    Latency_Can_Pending(0u);
    Latency_Capture_Push(&record);
}

static FDCAN_HandleTypeDef *handles[CAN_BUS_NUM] = { &hfdcan1, &hfdcan2, &hfdcan3 };


/* 按总线号取句柄 */
FDCAN_HandleTypeDef *Can_Bus_Handle(uint8_t bus)
{
    if (bus < 1u || bus > CAN_BUS_NUM)
    {
        return NULL;
    }
    return handles[bus - 1u];
}

/* 收帧计数 */
uint32_t Can_Bus_Rx_Count(uint8_t bus)
{
    if (bus < 1u || bus > CAN_BUS_NUM)
    {
        return 0u;
    }
    return can_bus[bus - 1u].alive_cnt;
}

/* 最近一帧的 ID */
uint32_t Can_Bus_Last_Rx_Id(uint8_t bus)
{
    if (bus < 1u || bus > CAN_BUS_NUM)
    {
        return 0u;
    }
    return can_bus[bus - 1u].last_rx_id;
}

/* 句柄指针→下标 */
static int32_t Can_Bus_Hw_Index(const FDCAN_HandleTypeDef *hfdcan)
{
    if (hfdcan == &hfdcan1) return 0;
    if (hfdcan == &hfdcan2) return 1;
    if (hfdcan == &hfdcan3) return 2;
    return -1;
}

/* 取消TX + 停止 */
static void Can_Bus_Stop(uint32_t idx)
{
    FDCAN_HandleTypeDef *hfdcan = handles[idx];
    uint32_t mask = __get_PRIMASK();
    __disable_irq();
    for (uint32_t slot = 0u; slot < 32u; slot++)
    {
        uint32_t bit = 1u << slot;
        uint8_t completed = (uint8_t)((hfdcan->Instance->TXBTO & bit) && !(hfdcan->Instance->TXBRP & bit));
        Can_Latency_Finish(idx, slot, Mono_Ns_Get(), completed ? 2u : 4u);
    }
    __set_PRIMASK(mask);
    SET_BIT(hfdcan->Instance->TXBCR, hfdcan->Instance->TXBRP);
    HAL_FDCAN_Stop(hfdcan);
}

/* 配置滤波器 + 启动 + 开中断 */
static bool Can_Bus_Start(uint32_t idx)
{
    FDCAN_HandleTypeDef *hfdcan = handles[idx];

    FDCAN_FilterTypeDef filter = {0};
    filter.IdType       = FDCAN_STANDARD_ID;
    filter.FilterType   = FDCAN_FILTER_MASK;
    filter.FilterConfig = FDCAN_FILTER_TO_RXFIFO0;
    filter.FilterID1    = 0x000;
    filter.FilterID2    = 0x000;
    filter.FilterIndex = 0;
    if (HAL_FDCAN_ConfigFilter(hfdcan, &filter) != HAL_OK || HAL_FDCAN_Start(hfdcan) != HAL_OK)
    {
        return false;
    }
    if (HAL_FDCAN_ActivateNotification(hfdcan, FDCAN_IT_RX_FIFO0_NEW_MESSAGE, 0) != HAL_OK || HAL_FDCAN_ActivateNotification(hfdcan, FDCAN_IT_RX_FIFO0_MESSAGE_LOST, 0) != HAL_OK || HAL_FDCAN_ActivateNotification(hfdcan, FDCAN_IT_BUS_OFF, 0) != HAL_OK)
    {
        (void)HAL_FDCAN_Stop(hfdcan);
        return false;
    }
    /* TX事件FIFO仍关闭；只观察硬件完成位，不改CAN速率、优先级或报文。 */
    (void)HAL_FDCAN_ActivateNotification(hfdcan, FDCAN_IT_TX_COMPLETE, 0xffffffffu);
    return true;
}

/* 初始化 */
void Can_Bus_Init(void)
{
    for (uint32_t i = 0; i < CAN_BUS_NUM; i++)
    {
        can_bus_t *bus = &can_bus[i];

        bus->tx_template.Identifier          = 0;
        bus->tx_template.IdType              = FDCAN_STANDARD_ID;
        bus->tx_template.TxFrameType         = FDCAN_DATA_FRAME;
        bus->tx_template.DataLength          = FDCAN_DLC_BYTES_8;
        bus->tx_template.ErrorStateIndicator = FDCAN_ESI_ACTIVE;
        bus->tx_template.BitRateSwitch       = FDCAN_BRS_OFF;
        bus->tx_template.FDFormat            = FDCAN_CLASSIC_CAN;
        bus->tx_template.TxEventFifoControl  = FDCAN_NO_TX_EVENTS;
        bus->tx_template.MessageMarker       = 0;

        bus->state        = CAN_BUS_STATE_ACTIVE;
        bus->route_cnt    = 0;
        bus->alive_cnt    = 0;
        bus->alive_prev   = 0;
        bus->last_tx_tick = HAL_GetTick();
        bus->dead_since   = HAL_GetTick();
        bus->reinit_tick  = HAL_GetTick();


        bus->state = Can_Bus_Start(i) ? CAN_BUS_STATE_ACTIVE : CAN_BUS_STATE_DEAD;
    }
}

/* 路由注册 */
bool Can_Bus_Register(FDCAN_HandleTypeDef *hfdcan, uint32_t can_id,
                      can_parse_fn_t fn, void *ctx)
{
    int32_t idx = Can_Bus_Hw_Index(hfdcan);
    if (idx < 0) return false;

    can_bus_t *bus = &can_bus[idx];
    if (bus->route_cnt >= CAN_BUS_ROUTE_MAX) return false;

    for (uint32_t i = 0; i < bus->route_cnt; i++)
        if (bus->route[i].can_id == can_id) return false;

    uint32_t slot = bus->route_cnt;
    can_route_t *r = &bus->route[slot];
    r->can_id = can_id;
    r->parse  = fn;
    r->ctx    = ctx;
    __DMB();
    bus->route_cnt = slot + 1u;
    return true;
}

/* 发送 (兼容旧接口) */
HAL_StatusTypeDef Can_Bus_Transmit(FDCAN_HandleTypeDef *hfdcan, uint32_t can_id,
                                   const uint8_t *data, uint8_t len)
{
    FDCAN_TxHeaderTypeDef tx_header;
    HAL_StatusTypeDef status;
    latency_record_t record = {0};
    uint64_t returned_ns;
    uint32_t slot;
    int32_t idx;

    record.kind = LATENCY_CAN;
    record.begin_ns = Mono_Ns_Get();
    record.value[0] = can_id;
    idx = Can_Bus_Hw_Index(hfdcan);
    if (idx < 0 || len > 8u) { return HAL_ERROR; }
    record.source = (uint8_t)(idx + 1);
    memcpy(&record.value[4], data, len); /* 原报文，不重新量化/换算。 */
    Latency_Can_Context(&record);
    taskENTER_CRITICAL();
    record.correlation = ++latency_tx_sequence;
    if (HAL_FDCAN_GetTxFifoFreeLevel(hfdcan) == 0u)
    {
        status = HAL_ERROR;
        slot = UINT32_MAX;
    }
    else
    {
        slot = (hfdcan->Instance->TXFQS & FDCAN_TXFQS_TFQPI) >> FDCAN_TXFQS_TFQPI_Pos;
        if (latency_tx_active[idx][slot])
        {
            uint32_t bit = 1u << slot;
            uint8_t completed = (uint8_t)((hfdcan->Instance->TXBTO & bit) && !(hfdcan->Instance->TXBRP & bit));
            Can_Latency_Finish((uint32_t)idx, slot, Mono_Ns_Get(), completed ? 2u : 4u);
        }
        tx_header = can_bus[idx].tx_template;
        tx_header.Identifier = can_id;
        tx_header.DataLength = len;
        status = HAL_FDCAN_AddMessageToTxFifoQ(hfdcan, &tx_header, (uint8_t *)data);
    }
    returned_ns = Mono_Ns_Get();
    record.value[2] = (uint32_t)returned_ns;
    record.value[3] = (uint32_t)(returned_ns >> 32);
    if (status == HAL_OK)
    {
        can_bus[idx].last_tx_tick = HAL_GetTick();
        record.value[1] = slot;
        latency_tx[idx][slot] = record;
        latency_tx_active[idx][slot] = 1u;
        Latency_Can_Pending(1u);
    }
    else
    {
        record.end_ns = returned_ns;
        record.flags = (uint8_t)((record.flags & 0x80u) | 3u);
        record.value[1] = (uint32_t)status;
        Latency_Capture_Push(&record);
    }
    taskEXIT_CRITICAL();
    return status;
}

/* TXBTO可能包含历史位；只消费仍有原提交上下文且不再pending的buffer。
 * 同一次IRQ的多个完成共享观测时刻，只能作完成延迟上界，不虚构物理完成次序。 */
void HAL_FDCAN_TxBufferCompleteCallback(FDCAN_HandleTypeDef *hfdcan, uint32_t BufferIndexes)
{
    int32_t idx = Can_Bus_Hw_Index(hfdcan);
    uint64_t now_ns = Mono_Ns_Get();
    if (idx < 0) { return; }
    for (uint32_t slot = 0u; slot < 32u; slot++)
    {
        uint32_t bit = 1u << slot;
        if ((BufferIndexes & bit) && !(hfdcan->Instance->TXBRP & bit))
        { Can_Latency_Finish((uint32_t)idx, slot, now_ns, 1u); }
    }
}


/* ISR 接收 */
void HAL_FDCAN_RxFifo0Callback(FDCAN_HandleTypeDef *hfdcan, uint32_t RxFifo0ITs)
{
    if (!(RxFifo0ITs & FDCAN_IT_RX_FIFO0_NEW_MESSAGE)) return;

    int32_t idx = Can_Bus_Hw_Index(hfdcan);
    if (idx < 0) return;

    can_bus_t *bus = &can_bus[idx];
    FDCAN_RxHeaderTypeDef rx_header;
    uint8_t data[64];

    while (HAL_FDCAN_GetRxFifoFillLevel(hfdcan, FDCAN_RX_FIFO0) > 0)
    {
        if (HAL_FDCAN_GetRxMessage(hfdcan, FDCAN_RX_FIFO0, &rx_header, data) != HAL_OK)
            break;

        bus->alive_cnt++;
        bus->last_rx_id = rx_header.Identifier;
        /* FD 帧也放行 (达妙 FD 1M/4M 反馈) */
        if (rx_header.IdType != FDCAN_STANDARD_ID || rx_header.RxFrameType != FDCAN_DATA_FRAME || rx_header.DataLength != FDCAN_DLC_BYTES_8)
        {
            continue;
        }

        for (uint32_t i = 0; i < bus->route_cnt; i++)
        {
            if (bus->route[i].can_id == rx_header.Identifier)
            {
                bus->route[i].parse(bus->route[i].ctx,
                                    rx_header.Identifier, data,
                                    (uint8_t)rx_header.DataLength);
                break;
            }
        }
    }
}

/* ISR 错误: bus-off → 停止 + 标记DEAD */
void HAL_FDCAN_ErrorStatusCallback(FDCAN_HandleTypeDef *hfdcan, uint32_t ErrorStatusITs)
{
    if (!(ErrorStatusITs & FDCAN_IT_BUS_OFF)) return;

    int32_t idx = Can_Bus_Hw_Index(hfdcan);
    if (idx < 0) return;

    Can_Bus_Stop(idx);
    can_bus[idx].state = CAN_BUS_STATE_DEAD;
}


/* 看门狗 */
bool Can_Bus_Online(bool expect_traffic)
{
    uint32_t now = HAL_GetTick();
    bool all_ok = true;

    for (uint32_t i = 0; i < CAN_BUS_NUM; i++)
    {
        can_bus_t *bus = &can_bus[i];

        /* 本机没在这条总线登记设备 → 不要求流量 */
        if (bus->route_cnt == 0u)
        {
            bus->alive_prev = bus->alive_cnt;
            bus->dead_since = now;
            continue;
        }

        bool alive_changed = (bus->alive_cnt != bus->alive_prev);

        /* DEAD + 收到新帧 → 恢复 */
        if (bus->state == CAN_BUS_STATE_DEAD && alive_changed)
            bus->state = CAN_BUS_STATE_ACTIVE;

        /* DEAD → 每100ms重新启动 */
        if (bus->state == CAN_BUS_STATE_DEAD && now - bus->reinit_tick >= CAN_BUS_REINIT_MS)
        {
            Can_Bus_Stop(i);
            (void)Can_Bus_Start(i);
            bus->reinit_tick = now;
        }

        if (!expect_traffic)
        {
            bus->alive_prev = bus->alive_cnt;
            bus->dead_since = now;
            if (bus->state == CAN_BUS_STATE_DEAD)
                all_ok = false;
            continue;
        }

        /* RX 看门狗 */
        if (alive_changed)
        {
            bus->alive_prev = bus->alive_cnt;
            bus->dead_since = now;
        }
        else if (now - bus->dead_since > CAN_BUS_DEAD_MS)
        {
            if (bus->state == CAN_BUS_STATE_ACTIVE)
                bus->state = CAN_BUS_STATE_DEAD;
        }

        if (bus->state == CAN_BUS_STATE_DEAD)
            all_ok = false;
    }
    return all_ok;
}

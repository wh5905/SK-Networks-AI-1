import { ActionContext } from "vuex"
import { OrderItem, OrderState } from "./states"
import { AxiosResponse } from "axios"
import axiosInst from "@/utility/axiosInstance"

export type OrderActions = {
    requestCreateOrderToDjango(
        context: ActionContext<OrderState, any>,
        payload: {
            userToken: string;
            items: {
                cartItemsId: number;
                quantity: number;
                orderPrice: number
            }[]
        }
    ): Promise<AxiosResponse>;

    // requestReadOrderToDjango(
    //     context: ActionContext<OrderState, any>,
    //     payload: {
    //         orderId: string
    //     }
    // ): Promise<AxiosResponse>
}

const actions: OrderActions = {
    async requestCreateOrderToDjango({ state }, payload) {
        try {
            const userToken = localStorage.getItem('userToken');
            if (!userToken) {
                throw new Error('User token not found');
            }

            console.log('payload:', payload)

            const requestData = {
                userToken,
                items: payload.items.map(item => ({
                    cartItemId: item.cartItemsId,
                    quantity: item.quantity,
                    orderPrice: item.orderPrice
                }))
            };
            console.log(requestData.items)
            const response =
                await axiosInst.djangoAxiosInst.post('/orders/create', requestData);
            console.log('response data:', response.data)

            return response.data;
        } catch (error) {
            console.error('Error creating order:', error);
            throw error;
        }
    },
    // async requestReadOrderToDjango({ state }, payload: { orderId: string }) {
        
    //     try {
    //         const { orderId } = payload
    //         const response = await axiosInst.djangoAxiosInst.post(`/orders/read/${orderId}`, requestData)
    //         return response.data
    //     } catch (error) {
    //         console.error('Error reading order:', error)
    //     }
    // }
};

export default actions;
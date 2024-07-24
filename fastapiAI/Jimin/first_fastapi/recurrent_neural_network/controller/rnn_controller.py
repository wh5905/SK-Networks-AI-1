from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from recurrent_neural_network.service.rnn_service_impl import RecurrentNeuralNetworkServiceImpl

recurrentNeuralNetworkRouter = APIRouter()

async def injectRecurrentNeuralNetworkService() -> RecurrentNeuralNetworkServiceImpl:
    return RecurrentNeuralNetworkServiceImpl()

# 흐름보기 위한 라이트버전
@recurrentNeuralNetworkRouter.post("/rnn-train")
async def rnnBasedTextTrain(recurrentNeuralNetworkService: RecurrentNeuralNetworkServiceImpl =
                            Depends(injectRecurrentNeuralNetworkService)):

    print(f"controller -> rnnBasedTextTrain()")

    recurrentNeuralNetworkService.trainText()

class RnnRequestForm(BaseModel):
    inputText: str

@recurrentNeuralNetworkRouter.post("/rnn-predict")
async def rnnBasedTextPredict(rnnRequestForm: RnnRequestForm,
                              recurrentNeuralNetworkService: RecurrentNeuralNetworkServiceImpl =
                            Depends(injectRecurrentNeuralNetworkService)):

    inputText = rnnRequestForm.inputText


    if not inputText:
        raise HTTPException(status_code=400, detail='텍스트 입력을 해주세요!')

    print(f"controller -> rnnBasedTextPredict()")

    # 현재 상황에선 매우 형편 없을 것으로 기대됨
    generatedText = recurrentNeuralNetworkService.predictText(inputText)
    return { "generatedText": generatedText }

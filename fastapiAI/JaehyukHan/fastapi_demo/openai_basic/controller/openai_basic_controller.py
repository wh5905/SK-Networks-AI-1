from fastapi import APIRouter, Depends, status, File, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

from openai_basic.controller.request_form.openai_basic_request_form import OpenAITalkRequestForm
from openai_basic.controller.request_form.openai_paper_similarity_analysis_request_form import \
    OpenAIPaperSimilarityAnalysisRequestForm
from openai_basic.service.openai_basic_service_impl import OpenAIBasicServiceImpl

openAIBasicRouter = APIRouter()

async def injectOpenAIBasicService() -> OpenAIBasicServiceImpl:
    return OpenAIBasicServiceImpl()

@openAIBasicRouter.post('/lets-talk')
async def talkWithOpenAI(openAITalkRequestForm: OpenAITalkRequestForm,
                         openAIBasicService: OpenAIBasicServiceImpl =
                         Depends(injectOpenAIBasicService)):
    print(f"controller -> talkWithOpenAI()\nopenAITalkRequestForm: {openAITalkRequestForm}")

    openAIGeneratedText = await openAIBasicService.letsTalk(openAITalkRequestForm.userSendMessage)

    return JSONResponse(content=openAIGeneratedText, status_code=status.HTTP_200_OK)


@openAIBasicRouter.post("/openai-audio")
async def audioAnalysisWithOpenAI(file: UploadFile = File(...),
                                  openAIBasicService: OpenAIBasicServiceImpl =
                                  Depends(injectOpenAIBasicService)):
    print(f"controller -> audioAnalysisWithOpenAI(): file: {file}")

    analyzedAudio = await openAIBasicService.audioAnalysis(file)

    return JSONResponse(content=analyzedAudio, status_code=status.HTTP_200_OK)

@openAIBasicRouter.post("/openai-sentiment")
async def sentimentAnalysisWithOpenAI(openAITalkRequestForm: OpenAITalkRequestForm,
                         openAIBasicService: OpenAIBasicServiceImpl =
                         Depends(injectOpenAIBasicService)):

    print(f"controller -> sentimentAnalysisWithOpenAI(): openAITalkRequestForm: {openAITalkRequestForm}")

    analyzedSentiment = await openAIBasicService.sentimentAnalysis(openAITalkRequestForm.userSendMessage)

    return JSONResponse(content=analyzedSentiment, status_code=status.HTTP_200_OK)

@openAIBasicRouter.post("/openai-audio")
async def audioAnalysisWithOpenAI(file: UploadFile = File(...),
                                  openAIBasicService: OpenAIBasicServiceImpl =
                                  Depends(injectOpenAIBasicService)):

    print(f"controller -> audioAnalysisWithOpenAI(): file: {file}")

    analyzedAudio = await openAIBasicService.audioAnalysis(file)

    return JSONResponse(content=analyzedAudio, status_code=status.HTTP_200_OK)

@openAIBasicRouter.post("/openai-similarity-analysis")
async def textSimilarityAnalysisWithOpenAI(
        openAIPaperSimilarityAnalysisRequestForm: OpenAIPaperSimilarityAnalysisRequestForm,
        openAIBasicService: OpenAIBasicServiceImpl =
        Depends(injectOpenAIBasicService)):

    print(f"controller -> textSimilarityAnalysisWithOpenAI():\n"
          f"openAITalkRequestForm: {openAIPaperSimilarityAnalysisRequestForm}")

    analyzedSimilarityText = await openAIBasicService.textSimilarityAnalysis(
        openAIPaperSimilarityAnalysisRequestForm.paperTitleList,
        openAIPaperSimilarityAnalysisRequestForm.userRequestPaperTitle)

    return JSONResponse(
        content={ "result": jsonable_encoder(analyzedSimilarityText) },
        status_code=status.HTTP_200_OK)

import os

import aiomysql
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from async_db.database import getMySqlPool, createTableIfNeccessary
from decision_tree.controller.decision_tree_controller import decisionTreeRouter
from exponential_regression.controller.exponential_regression_controller import exponentialRegressionRouter
from gradient_descent.controller.gradient_descent_controller import gradientDescentRouter
from kmeans.controller.keans_controller import kmeansRouter
from post.controller.post_controller import postRouter
from random_forest.controller.random_forest_controller import randomForestRouter
from logistic_regression.controller.logistic_regression_controller import logisticRegressionRouter
from polynomialRegression.controller.polynomial_regression_controller import polynomialRegressionRouter
from tf_iris.controller.tf_iris_controller import tfIrisRouter
from train_test_evaluation.controller.train_test_evaluation_controller import trainTestEvaluationRouter

import warnings
warnings.filterwarnings('ignore', category=aiomysql.Warning)

# 현재는 deprecated라고 나타나지만 lifespan이란 것을 대신 사용하라고 나타나고 있음
# 완전히 배제되지는 않았는데, 애플리케이션이 시작할 떄 실행될 함수를 지정함
# 고로 애플리케이션 시작시 비동기 처리가 가능한 DB를 구성한다고 보면 됨

# @app.on_event("startup")
# async def startup_event():
#     app.state.db_pool = await getMySqlPool()
#     await createTableIfNeccessary(app.state.db_pool)


# 위의 것이 킬 때 였으니 이건 반대라 보면 됨
# @app.on_event("shutdown")
# async def shutdown_event():
#     app.state.db_pool.close()
#     await app.state.db_pool.wait_closed()
async def lifespan(app: FastAPI):
    # Startup
    app.state.dbPool = await getMySqlPool()
    await createTableIfNeccessary(app.state.dbPool)

    yield

    # Shutdown
    app.state.dbPool.close()
    await app.state.dbPool.wait_closed()


app = FastAPI(lifespan=lifespan)

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "q": q}

app.include_router(logisticRegressionRouter)
app.include_router(trainTestEvaluationRouter)
app.include_router(polynomialRegressionRouter)
app.include_router(exponentialRegressionRouter)
app.include_router(randomForestRouter)
app.include_router(postRouter, prefix='/post')
app.include_router(kmeansRouter)
app.include_router(tfIrisRouter)
app.include_router(gradientDescentRouter)
app.include_router(decisionTreeRouter)

load_dotenv()

origins = os.getenv("ALLOWED_ORIGINS", "").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=33333)
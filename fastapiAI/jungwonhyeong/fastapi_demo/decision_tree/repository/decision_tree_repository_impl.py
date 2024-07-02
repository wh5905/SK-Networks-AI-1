import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from decision_tree.repository.decision_tree_repository import DecisionTreeRepository
import tensorflow as tf
# pip install tensorflow_decision_forests
# pip install --upgrade typing-extensions
import tensorflow_decision_forests as tfdf


class DecisionTreeRepositoryImpl(DecisionTreeRepository):

    def loadWineInfo(self):
        return load_wine()

    def createDataFrame(self, data, featureNames):
        df = pd.DataFrame(data=data, columns=featureNames)
        return df

    def splitTrainTestSet(self, wineDataFrame):
        return train_test_split(wineDataFrame, test_size=0.2, random_state=42)

    def applyStandardscaler(self, trainDataFrame, testDataFrame, featureNames):
        scaler = StandardScaler()

        # fit: trainDataFrame의 각 특성에 대해 평균과 표준편차를 계산합니다.
        # transform: 계산된 평균과 표준편차를 사용하여 trainDataFrame의 각 특성을 표준화합니다.
        # testDataFrame[featureNames] = scaler.transform(testDataFrame[featureNames])는 테스트 데이터셋을 표준화합니다.
        # 여기서 주의할 점은 테스트 데이터셋을 표준화할 때도 훈련 데이터셋의 평균과 표준편차를 사용한다는 것입니다.
        # 이는 테스트 데이터셋이 훈련 데이터셋과 동일한 분포를 가지도록 보장하기 위함입니다.

        trainDataFrame[featureNames] = scaler.fit_transform(trainDataFrame[featureNames])
        testDataFrame[featureNames] = scaler.transform(testDataFrame[featureNames])

        return trainDataFrame, testDataFrame

    def sliceTensor(self, scaledTrainDataFrame, scaledTestDataFrame):
        trainDataFrameAfterSlice = tf.data.Dataset.from_tensor_slices(
            (dict(scaledTrainDataFrame.drop("target", axis=1)),
            scaledTrainDataFrame['target'].astype(int))
        )
        testDataFrameAfterSlice = tf.data.Dataset.from_tensor_slices(
            (dict(scaledTestDataFrame.drop("target", axis=1)),
            scaledTestDataFrame['target'].astype(int))
        )

        return trainDataFrameAfterSlice, testDataFrameAfterSlice

    def applyBatchSize(self, trainDataFrameAfterSlice, testDataFrameAfterSlice, batchSize):
        readyForLearnTrainData = trainDataFrameAfterSlice.batch(batchSize)
        readyForLearnTestData = testDataFrameAfterSlice.batch(batchSize)

        return readyForLearnTrainData, readyForLearnTestData


    def learn(self, readyForLearnTrainData):
        model = tfdf.keras.RandomForestModel(task=tfdf.keras.Task.REGRESSION)
        model.fit(readyForLearnTrainData)
        return model
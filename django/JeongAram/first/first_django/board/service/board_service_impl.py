from board.service.board_service import BoardService
from board.repository.board_repository_impl import BoardRepositoryImpl


class BoardServiceImpl(BoardService):
    __instance = None

    def __new__(cls):
        if cls.__instance is None:
            cls.__instance = super().__new__(cls)
            cls.__instance.__boardRepository = BoardRepositoryImpl.getInstance()

        return cls.__instance

    @classmethod
    def getInstance(cls):
        if cls.__instance is None:
            cls.__instance = cls()

        return cls.__instance

    def list(self):
        self.__boardRepository.list()

    def createBoard(self, boardData):
        self.__boardRepository.create(boardData)

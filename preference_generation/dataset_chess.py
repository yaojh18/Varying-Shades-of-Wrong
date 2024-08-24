import argparse
import chess
from chess import Board, Move
from chess.engine import SimpleEngine
from datasets import load_dataset
from preference_generation.utils import *


class ChessPuzzle(RawPreferenceDataset):

    def __init__(self, **kwargs):
        self.output_name = kwargs['dataset_name']
        self.STOCKFISH_PATH = '../model/stockfish_windows/stockfish-windows-x86-64-avx2.exe'
        self.extract_pattern = r'([A-Z])(\.|\. .+)?$'
        self.map_into_index = False
        super().__init__(**kwargs)

    def load_dataset(self):
        raw_dataset = load_dataset('lczerolens/lichess-puzzles')['train'][:100000]
        for i in range(100000):
            if 600 < raw_dataset['Rating'][i] < 1000:
                board = Board(raw_dataset['FEN'][i])
                moves = raw_dataset['Moves'][i].split(' ')
                if len(moves) > 2:
                    board.push(Move.from_uci(moves[0]))
                    self.dataset.append({
                        'FEN': board.fen(),
                        'correct_answer': board.san(Move.from_uci(moves[1]))
                    })

    def precess_dataset(self, sample_size):
        if 0 < sample_size < len(self.dataset):
            random.seed(42)
            self.dataset = random.sample(self.dataset, sample_size)
        instruction = "You are a master-level chess player. You will be given a chess position in Forsyth-Edwards Notation (FEN) format. Your task is to analyze the position and choose the best move in Universal Chess Interface (UCI) format for the current player. Note that the move should improve the current player's position by considering both immediate benefits and long-term strategies."
        engine = SimpleEngine.popen_uci(self.STOCKFISH_PATH)
        for data in self.train_dataset:
            query = f"Instruction: {instruction}\nFEN: {data['FEN']}\nOptions: "
            board = Board(data['FEN'])
            info = engine.analyse(board, chess.engine.Limit(time=2.0), multipv=4)
            options = []
            original_correctness = []
            for move_info in info:
                move = move_info["pv"][0]
                score = move_info["score"].relative
                if score.is_mate():
                    if score.mate() > 0:
                        win_chance = 1.0
                    else:
                        win_chance = 0.0
                else:
                    cp = score.score()
                    win_chance = 1 / (1 + 10 ** (-cp / 400))
                options.append(board.san(move))
                original_correctness.append(win_chance)
            choices = []
            correctness = []
            for i in range(self.response_sample_size):
                sampled_idxs = random.sample(range(len(options)), len(options))
                choice = ''
                for j in range(len(options)):
                    choice += f"{idx2letter[j]}. {options[sampled_idxs[j]]} "
                correctness.append([original_correctness[idx] for idx in sampled_idxs])
                choices.append(choice)
            data['query'] = query + '\n'
            data['choices'] = choices
            data['correctness'] = correctness
            del data['FEN']
        engine.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate and save answers for ChessPuzzle dataset')
    parser.add_argument('--dataset_name', type=str, default='ChessPuzzle', help='Name of the dataset')
    parser.add_argument('--model_name', type=str, default='llama-3', help='Name of the model')
    parser.add_argument('--instruction_name', type=str, default='CoT',
                        help='Name of the instruction for generating answers')
    parser.add_argument('--extract_instruction_name', type=str, default='multi_choice_extract',
                        help='Name of the instruction for extracting answers')
    parser.add_argument('--dataset_sample_size', type=int, default=625, help='Dataset sample size')
    parser.add_argument('--response_sample_size', type=int, default=10, help='Response sample size')
    parser.add_argument('--load_from_exist', type=bool, default=False, help='Load from existing dataset or not')

    args = parser.parse_args()
    chess_dataset = ChessPuzzle(
        dataset_name=args.dataset_name,
        model_name=args.model_name,
        dataset_sample_size=args.dataset_sample_size,
        response_sample_size=args.response_sample_size,
        load_from_exist=args.load_from_exist
    )
    chess_dataset.generate_answer(instruction_name=args.instruction_name)
    chess_dataset.process_answer(instruction_name=args.instruction_name, extract_instruction_name=args.extract_instruction_name)
    chess_dataset.save_dataset()

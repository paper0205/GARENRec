import pandas as pd
import torch
from torch.utils.data import Dataset

class DraftDatasetWithFeatures(Dataset):
    """
    밴픽 데이터셋 클래스 (매치 피처 포함)
    """
    def __init__(self, data, champion_to_idx, position_to_idx, processed_match_data):
        """
        Args:
            data (pd.DataFrame): 밴픽 데이터
            champion_to_idx (dict): 챔피언 이름을 인덱스로 매핑
            position_to_idx (dict): 포지션을 인덱스로 매핑
            processed_match_data (dict): 전처리된 매치 데이터
        """
        self.data = data
        self.champion_to_idx = champion_to_idx
        self.position_to_idx = position_to_idx
        self.processed_match_data = processed_match_data
        self.feature_dim = 18  # 기본 피처 14개 + 추가 피처 4개 (kda, gpm, vspm, damage_share)

        # 밴픽 순서 정의
        self.ban_steps = ['BB1','RB1','BB2','RB2','BB3','RB3',
                         'RB4','BB4','RB5','BB5']
        self.pick_steps = ['BP1','RP1','RP2','BP2','BP3','RP3',
                          'RP4','BP4','BP5','RP5']
        # 전체 드래프트 순서
        self.draft_order = self.ban_steps[:6] + self.pick_steps[:6] + \
                          self.ban_steps[6:] + self.pick_steps[6:]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        한 경기의 드래프트 데이터 반환

        Returns:
            dict: {
                'draft_sequence': 챔피언 인덱스 시퀀스,
                'position_sequence': 포지션 인덱스 시퀀스,
                'mask_sequence': 픽/밴 마스크 시퀀스,
                'feature_sequence': 매치 피처 시퀀스
            }
        """
        row = self.data.iloc[idx]
        draft_sequence = []    # 챔피언 인덱스 시퀀스
        position_sequence = [] # 포지션 인덱스 시퀀스
        mask_sequence = []     # 픽/밴 마스크 시퀀스
        feature_sequence = []  # 매치 피처 시퀀스

        pick_count = 0  # 픽 순서 추적용

        for step in self.draft_order:
            # 챔피언 정보 처리
            champ = row[step] if pd.notna(row[step]) else 'NO_BAN'
            champ_idx = self.champion_to_idx[champ]
            draft_sequence.append(champ_idx)

            # 픽/밴 여부 확인
            is_pick = step in self.pick_steps
            mask_sequence.append(1 if is_pick else 0)

            if is_pick:
                # 픽 단계 처리
                # 포지션 정보 (BR: Blue Role, RR: Red Role)
                pos_col = 'BR' + step[-1] if step.startswith('B') else 'RR' + step[-1]
                pos = row[pos_col] if pd.notna(row[pos_col]) else 'Top'
                position_sequence.append(self.position_to_idx[pos])

                # 매치 데이터에서 피처 추출
                team_side = 'Blue' if step.startswith('B') else 'Red'
                game_id = row['gameid']

                if (game_id in self.processed_match_data and
                    team_side in self.processed_match_data[game_id] and
                    pos in self.processed_match_data[game_id][team_side]):
                    # 매치 데이터가 있는 경우
                    features = list(self.processed_match_data[game_id][team_side][pos].values())
                    feature_sequence.append(features)
                else:
                    # 매치 데이터가 없는 경우
                    feature_sequence.append([0] * self.feature_dim)

                pick_count += 1
            else:
                # 밴 단계 처리
                feature_sequence.append([0] * self.feature_dim)

        # torch tensor로 변환
        return {
            'draft_sequence': torch.tensor(draft_sequence, dtype=torch.long),
            'position_sequence': torch.tensor(position_sequence, dtype=torch.long),
            'mask_sequence': torch.tensor(mask_sequence, dtype=torch.bool),
            'feature_sequence': torch.tensor(feature_sequence, dtype=torch.float)
        }

    def get_champion_name(self, idx):
        """챔피언 인덱스로 이름 조회"""
        for name, index in self.champion_to_idx.items():
            if index == idx:
                return name
        return None

    def get_position_name(self, idx):
        """포지션 인덱스로 이름 조회"""
        for name, index in self.position_to_idx.items():
            if index == idx:
                return name
        return None

def prepare_data(file_path, test_size=0.1, val_size=0.05, random_state=42):
    """
    데이터를 읽고 train/val/test로 분할
    포지션 정보가 없는 경기는 제외
    """
    print(f"Loading data from {file_path}")
    try:
        data = pd.read_csv(file_path)
        print(f"Initially loaded {len(data)} rows")

        # 포지션 컬럼 정의
        position_cols = ['BR1', 'BR2', 'BR3', 'BR4', 'BR5',
                        'RR1', 'RR2', 'RR3', 'RR4', 'RR5']

        # MISSING DATA가 있는 경기 ID 찾기
        missing_mask = data[position_cols].eq('MISSING DATA').any(axis=1)
        missing_games = data[missing_mask]['gameid'].unique()

        if len(missing_games) > 0:
            print(f"\nRemoving {len(missing_games)} games with missing position data:")
            print("Game IDs:", missing_games)

            # 해당 경기들 제외
            data = data[~data['gameid'].isin(missing_games)]
            print(f"Remaining rows after removal: {len(data)}")

        # 데이터가 비어있는지 확인
        if len(data) == 0:
            print("Error: The loaded data is empty")
            return None, None, None

        # 데이터 셔플
        data = data.sample(frac=1, random_state=random_state).reset_index(drop=True)

        # 데이터 분할
        total_size = len(data)
        test_count = int(total_size * test_size)
        val_count = int(total_size * val_size)

        test_data = data[:test_count]
        val_data = data[test_count:test_count+val_count]
        train_data = data[test_count+val_count:]

        print(f"\nSplit sizes:")
        print(f"Train: {len(train_data)}")
        print(f"Validation: {len(val_data)}")
        print(f"Test: {len(test_data)}")

        return train_data, val_data, test_data

    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None, None

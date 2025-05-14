import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import torch

class MatchDataProcessor:
    """
    매치 데이터 전처리 및 정규화 클래스
    """
    def __init__(self):
        self.feature_columns = [
            'kills', 'deaths', 'assists',
            'teamkills', 'teamdeaths',
            'goldat10', 'xpat10', 'csat10',
            'goldat15', 'xpat15', 'csat15',
            'damagetochampions', 'wardsplaced', 'visionscore'
        ]
        self.processed_data = None

    def preprocess_features(self, df):
        """
        매치 데이터 전처리 및 정규화
        """
        print("Preprocessing match data...")
        # 데이터 복사
        df = df.copy()

        # 1. KDA 계산 및 추가
        print("Calculating KDA...")
        df['kda'] = np.where(
            df['deaths'] == 0,
            df['kills'] + df['assists'],
            (df['kills'] + df['assists']) / df['deaths']
        )

        # 2. GPM (Gold Per Minute) 계산
        print("Calculating GPM...")
        df['gpm'] = df['goldat15'] / 15.0

        # 3. Vision Score Per Minute
        print("Calculating Vision Score Per Minute...")
        df['vspm'] = df['visionscore'] / 15.0

        # 4. 데미지 점유율 계산
        print("Calculating Damage Share...")
        df['damage_share'] = df.groupby('gameid')['damagetochampions'].transform(
            lambda x: x / x.sum()
        )

        # 5. 정규화
        print("Normalizing features...")
        # 5.1 KDA, GPM, VSPM - Standard Scaling
        for col in ['kda', 'gpm', 'vspm']:
            mean = df[col].mean()
            std = df[col].std()
            df[col] = (df[col] - mean) / (std + 1e-8)  # 0으로 나누는 것 방지

        # 5.2 Kill, Death, Assist - Min-Max Scaling
        for col in ['kills', 'deaths', 'assists']:
            min_val = df[col].min()
            max_val = df[col].max()
            df[col] = (df[col] - min_val) / ((max_val - min_val) + 1e-8)

        # 5.3 Gold, XP, CS differences - Robust Scaling
        diff_columns = ['golddiffat10', 'xpdiffat10', 'csdiffat10',
                       'golddiffat15', 'xpdiffat15', 'csdiffat15']
        for col in diff_columns:
            if col in df.columns:
                median = df[col].median()
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                df[col] = (df[col] - median) / (iqr + 1e-8)

        # 5.4 기타 수치형 feature columns의 Standard Scaling
        other_numeric = [col for col in self.feature_columns
                        if col not in ['kills', 'deaths', 'assists'] + diff_columns]
        for col in other_numeric:
            if col in df.columns:
                mean = df[col].mean()
                std = df[col].std()
                df[col] = (df[col] - mean) / (std + 1e-8)

        # 6. 결측치 처리
        print("Handling missing values...")
        df = df.fillna(0)

        return df

    def preprocess_once(self, match_df):
        """
        한 번만 실행되는 전처리
        """
        if self.processed_data is not None:
            return self.processed_data

        # 전처리 수행
        print("Starting data preprocessing...")
        processed_df = self.preprocess_features(match_df)

        # 처리된 데이터를 딕셔너리 형태로 변환하여 캐싱
        self.processed_data = self._convert_to_lookup_dict(processed_df)

        return self.processed_data

    def _convert_to_lookup_dict(self, df):
        """
        빠른 조회를 위한 중첩 딕셔너리 생성
        """
        lookup_dict = {}
        for _, row in df.iterrows():
            game_id = row['gameid']
            side = row['side']
            position = row['position']

            if game_id not in lookup_dict:
                lookup_dict[game_id] = {'Blue': {}, 'Red': {}}

            feature_dict = {col: row[col] for col in self.feature_columns}
            # 추가 피처들
            feature_dict.update({
                'kda': row['kda'],
                'gpm': row['gpm'],
                'vspm': row['vspm'],
                'damage_share': row['damage_share']
            })

            lookup_dict[game_id][side][position] = feature_dict

        return lookup_dict

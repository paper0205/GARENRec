import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch

class ChampionRelationships:
    """
    챔피언 간의 관계를 분석하고 그래프 데이터를 생성하는 클래스
    """
    def __init__(self, data, champion_to_idx):
        self.data = data
        self.champion_to_idx = champion_to_idx
        self.win_rates = self._calculate_win_rates()
        self.relationships = self._build_relationships()

    def _calculate_win_rates(self):
        """승률과 시너지/카운터 관계 계산"""
        # 챔피언별 승률
        champion_wins = {}
        champion_games = {}
        # 챔피언 페어의 승률 (시너지)
        synergy_wins = {}
        synergy_games = {}
        # 챔피언 카운터 관계
        counter_wins = {}
        counter_games = {}

        for _, row in self.data.iterrows():
            blue_picks = [f'BP{i}' for i in range(1, 6)]
            red_picks = [f'RP{i}' for i in range(1, 6)]
            blue_team = [row[pick] for pick in blue_picks if pd.notna(row[pick])]
            red_team = [row[pick] for pick in red_picks if pd.notna(row[pick])]
            if 'Winner' in row:
                blue_win = (row['Winner'] == 1)
            else:
                continue  # 승패 정보가 없는 경우 건너뜀

            if blue_win is not None:  # 승패 정보가 있는 경우만
                # 개별 챔피언 승률 계산
                for champ in blue_team:
                    champion_games[champ] = champion_games.get(champ, 0) + 1
                    if blue_win:
                        champion_wins[champ] = champion_wins.get(champ, 0) + 1

                for champ in red_team:
                    champion_games[champ] = champion_games.get(champ, 0) + 1
                    if not blue_win:
                        champion_wins[champ] = champion_wins.get(champ, 0) + 1

                # 시너지 관계 계산
                for i in range(len(blue_team)):
                    for j in range(i + 1, len(blue_team)):
                        pair = tuple(sorted([blue_team[i], blue_team[j]]))
                        synergy_games[pair] = synergy_games.get(pair, 0) + 1
                        if blue_win:
                            synergy_wins[pair] = synergy_wins.get(pair, 0) + 1

                for i in range(len(red_team)):
                    for j in range(i + 1, len(red_team)):
                        pair = tuple(sorted([red_team[i], red_team[j]]))
                        synergy_games[pair] = synergy_games.get(pair, 0) + 1
                        if not blue_win:
                            synergy_wins[pair] = synergy_wins.get(pair, 0) + 1

                # 카운터 관계 계산
                for blue_champ in blue_team:
                    for red_champ in red_team:
                        counter_pair = (blue_champ, red_champ)
                        counter_games[counter_pair] = counter_games.get(counter_pair, 0) + 1
                        if blue_win:
                            counter_wins[counter_pair] = counter_wins.get(counter_pair, 0) + 1

        # 승률 계산
        win_rates = {
            'champion': {champ: champion_wins.get(champ, 0) / games
                        for champ, games in champion_games.items()},
            'synergy': {pair: synergy_wins.get(pair, 0) / games
                       for pair, games in synergy_games.items()
                       if games >= 3},  # 최소 3게임 이상
            'counter': {pair: counter_wins.get(pair, 0) / games
                       for pair, games in counter_games.items()
                       if games >= 3}  # 최소 3게임 이상
        }

        return win_rates

    def _calculate_relationship_weight(self, champ_a, champ_b, same_team=True):
        """두 챔피언 간의 관계 가중치 계산"""
        base_weight = 1.0 if same_team else -1.0

        if same_team:
            # 시너지 관계
            pair = tuple(sorted([champ_a, champ_b]))
            if pair in self.win_rates['synergy']:
                synergy_rate = self.win_rates['synergy'][pair]
                # 승률 50%를 기준으로 가중치 조정
                weight_adjustment = (synergy_rate - 0.5) * 2
                return base_weight * (1 + weight_adjustment)
        else:
            # 카운터 관계
            counter_pair = (champ_a, champ_b)
            if counter_pair in self.win_rates['counter']:
                counter_rate = self.win_rates['counter'][counter_pair]
                # 승률 50%를 기준으로 가중치 조정
                weight_adjustment = (counter_rate - 0.5) * 2
                return base_weight * (1 + weight_adjustment)

        return base_weight

    def _build_relationships(self):
        """챔피언 관계 구축"""
        relationships = []

        for _, row in self.data.iterrows():
            blue_picks = [f'BP{i}' for i in range(1, 6)]
            red_picks = [f'RP{i}' for i in range(1, 6)]

            # 블루팀 내 관계
            blue_team = [row[pick] for pick in blue_picks if pd.notna(row[pick])]
            for i in range(len(blue_team)):
                for j in range(i + 1, len(blue_team)):
                    weight = self._calculate_relationship_weight(
                        blue_team[i], blue_team[j], same_team=True)
                    relationships.append((
                        blue_team[i],
                        blue_team[j],
                        weight
                    ))

            # 레드팀 내 관계
            red_team = [row[pick] for pick in red_picks if pd.notna(row[pick])]
            for i in range(len(red_team)):
                for j in range(i + 1, len(red_team)):
                    weight = self._calculate_relationship_weight(
                        red_team[i], red_team[j], same_team=True)
                    relationships.append((
                        red_team[i],
                        red_team[j],
                        weight
                    ))

            # 팀 간 카운터 관계
            for blue_champ in blue_team:
                for red_champ in red_team:
                    weight = self._calculate_relationship_weight(
                        blue_champ, red_champ, same_team=False)
                    relationships.append((
                        blue_champ,
                        red_champ,
                        weight
                    ))

        return relationships

    def get_champion_win_rate(self, champion):
        """특정 챔피언의 승률 반환"""
        return self.win_rates['champion'].get(champion, 0.5)

    def get_synergy_win_rate(self, champ_a, champ_b):
        """두 챔피언의 시너지 승률 반환"""
        pair = tuple(sorted([champ_a, champ_b]))
        return self.win_rates['synergy'].get(pair, 0.5)

    def get_counter_win_rate(self, champ_a, champ_b):
        """챔피언 A의 챔피언 B에 대한 카운터 승률 반환"""
        return self.win_rates['counter'].get((champ_a, champ_b), 0.5)

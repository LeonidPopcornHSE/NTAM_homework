import pandas as pd
import numpy as np


class DataHandler:
    def __init__(self):
        self.next_handler = None
    
    def set_next(self, handler):
        self.next_handler = handler
        return handler
    
    def handle(self, df):
        if self.next_handler:
            return self.next_handler.handle(df)
        return df


class LoadHandler(DataHandler):
    def __init__(self, file_path):
        super().__init__()
        from hh_parser import HHDataProcessor
        self.parser = HHDataProcessor()
        self.file_path = file_path
    
    def handle(self, df):
        return self.next_handler.handle(self.parser.parse(self.file_path))


class FeatureHandler(DataHandler):
    def __init__(self):
        super().__init__()
        from hh_parser import HHDataProcessor
        self.parser = HHDataProcessor()
    
    def handle(self, df):
        return self.next_handler.handle(self.parser.prepare_features(df))


class PrepareHandler(DataHandler):
    def handle(self, df):
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        return df[numeric_cols]
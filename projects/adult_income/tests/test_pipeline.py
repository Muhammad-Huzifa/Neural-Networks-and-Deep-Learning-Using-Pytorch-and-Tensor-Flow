import tempfile
import unittest
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from adult_income.data import FEATURES, prepare_features
from adult_income.model import build_pipeline, predict_records

class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame=pd.DataFrame({"age":[22,48,27,55]*8,"fnlwgt":[10000,20000,30000,40000]*8,"education-num":[9,13,10,16]*8,"capital-gain":[0,2000,0,4000]*8,"capital-loss":[0]*32,"hours-per-week":[30,50,35,55]*8,"workclass":["Private","Self-emp","Private","Private"]*8,"education":["HS-grad","Bachelors","Some-college","Masters"]*8,"marital-status":["Never-married","Married","Never-married","Married"]*8,"occupation":["Sales","Prof-specialty","Sales","Prof-specialty"]*8,"relationship":["Not-in-family","Husband","Not-in-family","Wife"]*8,"race":["White"]*32,"sex":["Male","Female"]*16,"country":["United-States"]*32})
        cls.pipeline=build_pipeline("logistic")
        cls.pipeline.fit(prepare_features(cls.frame),np.array([0,1,0,1]*8))

    def test_single_row_matches_batch_prediction(self):
        batch=predict_records(self.pipeline,self.frame.iloc[:4])
        for i in range(4):
            single=predict_records(self.pipeline,self.frame.iloc[[i]])[0]
            self.assertEqual(single['prediction'],batch[i]['prediction'])
            self.assertAlmostEqual(single['probability_gt_50k'],batch[i]['probability_gt_50k'])

    def test_unseen_category_uses_training_feature_layout(self):
        frame=self.frame.iloc[[0]].copy()
        frame['country']='Unseen-country'
        probability=predict_records(self.pipeline,frame)[0]['probability_gt_50k']
        self.assertTrue(0<=probability<=1)

    def test_persisted_pipeline_keeps_predictions(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'pipeline.joblib'
            joblib.dump(self.pipeline,path)
            self.assertEqual(predict_records(joblib.load(path),self.frame.iloc[:2]),predict_records(self.pipeline,self.frame.iloc[:2]))

    def test_missing_feature_is_reported(self):
        with self.assertRaisesRegex(ValueError,'hours-per-week'):
            prepare_features(self.frame.drop(columns='hours-per-week'))

    def test_input_aliases_and_spaces_are_normalized(self):
        frame=self.frame.iloc[[0]].rename(columns={'education-num':'education_num','country':'native-country'})
        frame['workclass']=' Private '
        self.assertEqual(prepare_features(frame)['workclass'].iloc[0],'Private')
        self.assertEqual(list(prepare_features(frame).columns),FEATURES)

if __name__=='__main__':
    unittest.main()

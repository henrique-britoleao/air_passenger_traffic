class ModelFeatureSelector():
    '''
    Performs Feature Selection based on one of the emsemble tree
    models of ScikitLearn. Higly correlated variables should have
    aleready been deleted. 
    '''
    def __init__(regressor, alpha):
        self.regressor = regressor
        self.alpha = alpha
    def fit(X, y):
        #fit regressor
        self.regressor.fit(X, y)
        #recover feat_importances
        feat_imp = self.regressor.feature_importances_
        feat = X.columns
        #store feat_import in a DataFrame
        res_df = pd.DataFrame(
            {'Features': feat, 'Importance': feat_imp}
        ).sort_values(by='Importance', ascending=False)
        #Calculate cumulative importance
        res_df.reset_index(implace=True)
        res_df.drop(columns=['index'], inplace=True)
        res_df['c_import'] = res_df.Importance.cumsum()
        self.not_import_feat = res_df.loc[
            res_df.c_import > 0.99, 'Features'
        ] # Select features that collectively contribute to
          # to less than 1% of the variance
    def transform(X):
        X.drop(columns=list(self.not_import_feat), inplace=True)
        
import numpy as np
import pytest
from reports.analyze_datasets import adjusted_r2, intervals_for_pipeline


def test_adjusted_r2_uses_residual_degrees_of_freedom():
    assert adjusted_r2(0.8, 100, 4) == pytest.approx(1 - .2*99/95)
    assert np.isnan(adjusted_r2(.8, 5, 4))


def test_prediction_interval_covers_mean_interval():
    import pandas as pd
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import Pipeline
    x=pd.DataFrame({'a':np.arange(12.),'b':np.arange(12.)**2})
    y=1+2*x.a-.1*x.b+np.sin(x.a)
    model=Pipeline([('model',LinearRegression())]).fit(x,y)
    frame=intervals_for_pipeline(model,x,y,x)
    assert np.allclose(frame['mean'],model.predict(x))
    assert ((frame.obs_ci_upper-frame.obs_ci_lower)>=(frame.mean_ci_upper-frame.mean_ci_lower)).all()

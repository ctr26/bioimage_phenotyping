import pytest
from bioimage_phenotyping import Cellprofiler

class TestData:
    
    @pytest.fixture(autouse=True)
    def setup(self):
        self.data_folder = "analysed/_2019_cellesce_unet_splineparameters_aligned/raw/projection_XY/"
        self.data = Cellprofiler(data_folder=self.data_folder)

    def data_construct_test(self):
        return Cellprofiler(data_folder=self.data_folder)
    
    def test_data_construct_test(self):
        return self.data_construct_test()
    
    def test_one(self):
            self.value = 1
            assert self.value == 1
# %%

# Cellprofiler("analysed/_2019_cellesce_unet_splineparameters_aligned/raw/projection_XY/")
# %%

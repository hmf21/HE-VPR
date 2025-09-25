from dataloaders_for_HE.he_test_datasets.MHFlightHeightDataset import MHFlightHEDataset
from dataloaders_for_HE.he_test_datasets.MHFlightiVPRDataset import MHFlightVPRDataset
from dataloaders_for_HE.he_test_datasets.GEStudioHeightDataset import GEStuidioHEDataset
from dataloaders_for_HE.he_test_datasets.GEStuidioVPRDataset import GEStuidioVPRDataset


def get_test_HEVPR_datasets(img_resize):
    Test_MHFlightVPRDataset = MHFlightVPRDataset(img_resize)
    Test_MHFlightHEDataset = MHFlightHEDataset(img_resize)
    Test_GEStuidioVPRDataset = GEStuidioVPRDataset(img_resize)
    Test_GEStuidioHEDataset = GEStuidioHEDataset(img_resize)
    return [Test_MHFlightVPRDataset, Test_MHFlightHEDataset, Test_GEStuidioVPRDataset, Test_GEStuidioHEDataset]

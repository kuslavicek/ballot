import numpy as np
import pytest
from ballot import utils

class TestCheckBalanceConstraints:
    def test_check_balance_constraints_valid_input(self):
        assert utils.check_balance_constraints(10, 2) is True
        assert utils.check_balance_constraints(9, 3) is True

    def test_check_balance_constraints_invalid_input_raises_error(self):
        with pytest.raises(ValueError, match="divisible"):
            utils.check_balance_constraints(10, 3)

class TestRoundTransportMatrix:
    def test_round_transport_matrix_preserves_marginals(self):
        n, k = 4, 2
        F = np.array([
            [0.25, 0],
            [0.25, 0],
            [0, 0.25],
            [0, 0.25]
        ])
        noise = np.random.normal(0, 0.001, (n, k))
        F_noisy = np.abs(F + noise)
        
        F_rounded = utils.round_transport_matrix(F_noisy, n, k)
        
        np.testing.assert_allclose(F_rounded.sum(axis=1), 1.0/n)
        np.testing.assert_allclose(F_rounded.sum(axis=0), 1.0/k)

    def test_round_transport_matrix_output_is_integral(self):
        n, k = 4, 2
        F_soft = np.full((n, k), 1.0/(n*k)) 
        
        F_rounded = utils.round_transport_matrix(F_soft, n, k)
        
        target = 1.0/n
        is_discrete = np.isclose(F_rounded, 0) | np.isclose(F_rounded, target)
        assert np.all(is_discrete)
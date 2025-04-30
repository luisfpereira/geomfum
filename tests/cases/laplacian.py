from geomstats.test.test_case import TestCase


class LaplacianFinderCmpCase(TestCase):
    """Laplacian finder comparison.

    Notes
    -----
    Needs: `finder_a`, `finder_b`
    """

    def test_matrices_cmp(self, shape, atol):
        laplacian_matrix, mass_matrix = self.finder_a(shape)
        laplacian_matrix_, mass_matrix_ = self.finder_b(shape)

        self.assertAllClose(
            laplacian_matrix.todense(), laplacian_matrix_.todense(), atol=atol
        )
        self.assertAllClose(mass_matrix.todense(), mass_matrix_.todense(), atol=atol)


class LaplacianSpectrumFinderCmpCase(TestCase):
    """Laplacian finder comparison.

    Notes
    -----
    Needs: `finder_a`, `finder_b`
    """

    def test_eigenvals_cmp(self, shape, atol):
        vals, _ = self.finder_a(shape, as_basis=False)
        vals_, _ = self.finder_b(shape, as_basis=False)

        self.assertAllClose(vals, vals_, atol=atol)

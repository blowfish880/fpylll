# -*- coding: utf-8 -*-

from random import randint
from math import floor
from fpylll import BKZ, Enumeration, EnumerationError
from fpylll.algorithms.bkz import BKZReduction as BKZBase
from fpylll.algorithms.bkz_stats import dummy_tracer
from fpylll.util import gaussian_heuristic


class BKZReduction(BKZBase):

    def __init__(self, A):
        """Create new BKZ object.

        :param A: an integer matrix, a GSO object or an LLL object

        """
        BKZBase.__init__(self, A)

    def get_pruning(self, kappa, block_size, param, tracer=dummy_tracer):
        strategy = param.strategies[block_size]

        radius, re = self.M.get_r_exp(kappa, kappa)
        root_det = self.M.get_root_det(kappa, kappa + block_size)
        gh_radius, ge = gaussian_heuristic(radius, re, block_size, root_det, 1.0)
        return strategy.get_pruning(radius  * 2**re, gh_radius * 2**ge)

    def randomize_block(self, min_row, max_row, tracer=dummy_tracer, density=0):
        """Randomize basis between from ``min_row`` and ``max_row`` (exclusive)

            1. permute rows

            2. apply lower triangular matrix with coefficients in -1,0,1

            3. LLL reduce result

        :param min_row: start in this row
        :param max_row: stop at this row (exclusive)
        :param tracer: object for maintaining statistics
        :param density: number of non-zero coefficients in lower triangular transformation matrix
        """
        if max_row - min_row < 2:
            return  # there is nothing to do

        # 1. permute rows
        niter = 4 * (max_row-min_row)  # some guestimate
        with self.M.row_ops(min_row, max_row):
            for i in range(niter):
                b = a = randint(min_row, max_row-1)
                while b == a:
                    b = randint(min_row, max_row-1)
                self.M.move_row(b, a)

        # 2. triangular transformation matrix with coefficients in -1,0,1
        with self.M.row_ops(min_row, max_row):
            for a in range(min_row, max_row-2):
                for i in range(density):
                    b = randint(a+1, max_row-1)
                    s = randint(0, 1)
                    self.M.row_addmul(a, b, 2*s-1)

        return

    def svp_preprocessing(self, kappa, block_size, param, tracer=dummy_tracer):
        clean = True

        clean &= BKZBase.svp_preprocessing(self, kappa, block_size, param, tracer)

        for preproc in param.strategies[block_size].preprocessing_block_sizes:
            prepar = param.__class__(block_size=preproc, strategies=param.strategies, flags=BKZ.GH_BND)
            clean &= self.tour(prepar, kappa, kappa + block_size)

        return clean

    def svp_reduction(self, kappa, block_size, param, tracer=dummy_tracer, dual=False):
        """

        :param kappa:
        :param block_size:
        :param params:
        :param tracer:

        """

        first = kappa + block_size - 1 if dual else kappa;
        self.lll_obj.size_reduction(0, first+1)
        old_first, old_first_expo = self.M.get_r_exp(first, first)

        remaining_probability, rerandomize = 1.0, False

        while remaining_probability > 1. - param.min_success_probability:
            with tracer.context("preprocessing"):
                if rerandomize:
                    with tracer.context("randomization"):
                        self.randomize_block(kappa+1, kappa+block_size,
                                             density=param.rerandomization_density, tracer=tracer)
                with tracer.context("reduction"):
                    self.svp_preprocessing(kappa, block_size, param, tracer=tracer)

            radius, expo = self.M.get_r_exp(first, first)
            if dual:
                radius = 1/radius
                expo *= -1
            radius *= self.lll_obj.delta

            if param.flags & BKZ.GH_BND and block_size > 30:
                #TODO: what to do if dual?
                root_det = self.M.get_root_det(kappa, kappa + block_size)
                radius, expo = gaussian_heuristic(radius, expo, block_size, root_det, param.gh_factor)

            pruning = self.get_pruning(kappa, block_size, param, tracer)

            try:
                enum_obj = Enumeration(self.M)
                with tracer.context("enumeration",
                                    enum_obj=enum_obj,
                                    probability=pruning.probability,
                                    full=block_size==param.block_size):
                    solution, max_dist = enum_obj.enumerate(kappa, kappa + block_size, radius, expo,
                                                            pruning=pruning.coefficients, dual=dual)[0]
                with tracer.context("postprocessing"):
                    if dual:
                        self.dsvp_postprocessing(kappa, block_size, solution)
                    else:
                        self.svp_postprocessing(kappa, block_size, solution, tracer=tracer)
                rerandomize = False

            except EnumerationError:
                rerandomize = True

            remaining_probability *= (1 - pruning.probability)

        self.lll_obj.size_reduction(0, first+1)
        new_first, new_first_expo = self.M.get_r_exp(first, first)

        if dual :
            clean = old_first >= new_first * 2**(new_first_expo - old_first_expo)
        else:
            clean = old_first <= new_first * 2**(new_first_expo - old_first_expo)
        return clean

    def euclid(self, pair1, pair2):
        """For input (x1,y1) and (x2,y2) compute the GCD of x1 and x2 while 
        also applying the operations on vectors y1 and y2.

        :param pair1: first input to GCD
        :param pair2: second input to GCD
        :returns: the GCD of x1 and x2 and the correspndinig 
        :rtype:

        """
        row1, x1 = pair1
        row2, x2 = pair2
        if not x1:
            return pair2
        c = floor(x2/x1)
        self.M.row_addmul(row2, row1, -c)
        return self.euclid((row2, x2 - c*x1), pair1)

    def dsvp_postprocessing(self, kappa, block_size, solution):
        """Insert DSVP solution into basis and LLL reduce.

        :param solution: coordinates of a DSVP solution
        :param kappa: current index
        :param block_size: block size
        :param stats: object for maintaining statistics

        :returns: ``True`` if no change was made and ``False`` otherwise
        """
        if solution is None:
            return True

        with self.M.row_ops(kappa, kappa+block_size):
            pairs = list(enumerate(solution, start=kappa))
            [self.M.negate_row(pair[0]) for pair in pairs if pair[1] < 0]
            pairs = map(lambda x: (x[0], abs(x[1])), pairs)
            # GCD should be tree based but for proof of concept implementation, this will do
            row, x = reduce(self.euclid, pairs)
            if x != 1:
                raise RuntimeError("Euclid failed!")
            self.M.move_row(row, kappa + block_size - 1)

        return False

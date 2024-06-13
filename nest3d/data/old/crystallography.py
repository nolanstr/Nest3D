import numpy as np

ZERO = 0
ONE = 1
TWO = 2
THREE = 3
FOUR = 4

ratioCOverA = 0
R2 = ONE / TWO
R3 = np.sqrt(THREE)

BUSH = ratioCOverA

# Rotation is ZXZ
EULER_ANGLES = np.array(
    [
        [135, 90, 354.736],
        [0, 135, 84.7356],
        [90, 45, 264.736],
        [315, 90, 174.736],
        [180, 45, 84.7356],
        [270, 135, 264.736],
        [0, 45, 264.736],
        [225, 90, 174.736],
        [180, 135, 264.736],
        [45, 90, 354.736],
        [90, 135, 84.7356],
        [270, 45, 84.7356],
    ]
)


class HCPVariants:

    """
    Crystal Type: 7, HCP Variants
    """

    def __init__(self):
        self.IC = 0
        self.number_of_variants = 12
        self.number_of_slip_systems = 30
        self.dimensions = 3
        self.directionsSlipSystem = np.empty(
            (self.dimensions, self.number_of_slip_systems)
        )  # HCP
        self.normalsSlipSystem = np.empty(
            (self.dimensions, self.number_of_slip_systems)
        )  # HCP

        self.store_basal_slip_system()
        self.store_prismatic_slip_system()
        self.store_pyramidal_A_slip()
        self.store_first_order_CpA_slip()
        self.store_second_order_CpA_slip()
        self.create_rotation_matrices()
        self.calculateMPrime()

    def compute_equivalent_diameters(self, modelData):
        if modelData.shape[0] % self.number_of_variants != 0:
            modelData = modelData[1:]

        modelData = modelData.reshape((-1, 3)).T
        equivalent_diameters = np.empty((self.number_of_variants, 30))
        allLathBasis = np.empty((self.number_of_variants, 3, 3))
        allShapeParameters = np.empty((self.number_of_variants, 3, 1))
        for i in range(self.number_of_variants):
            variantSlipDirections = np.hstack(
                [
                    np.matmul(self.Rs[i], refSlipDirection.reshape((-1, 1)))
                    for refSlipDirection in self.directionsSlipSystem.T
                ]
            )
            info = modelData[:, i * 4 : (i + 1) * 4]
            lathBasis = info[:, 0:-1]
            lathShapeParameters = info[:, -1] * 1e-6
            variant_equivalent_diameters = [
                self._compute_equivalent_diameter(
                    variantSlipDirection, lathBasis, lathShapeParameters
                )
                for variantSlipDirection in variantSlipDirections.T
            ]
            equivalent_diameters[i, :] = variant_equivalent_diameters

            allLathBasis[i] = lathBasis
            allShapeParameters[i] = lathShapeParameters.reshape((-1, 1))

        return allLathBasis, allShapeParameters, equivalent_diameters

    def _compute_equivalent_diameter(
        self, variantSlipDirection, lathBasis, lathShapeParameters
    ):
        term1 = np.square(
            lathShapeParameters[0] * variantSlipDirection.dot(lathBasis[:, 0])
        )
        term2 = np.square(
            lathShapeParameters[1] * variantSlipDirection.dot(lathBasis[:, 1])
        )
        term3 = np.square(
            lathShapeParameters[2] * variantSlipDirection.dot(lathBasis[:, 2])
        )
        equivalent_radius = np.sqrt(term1 + term2 + term3)
        equivalent_diameter = 2 * equivalent_radius

        return equivalent_diameter

    def create_rotation_matrices(self):
        """
        Performed in ZXZ rotation order
        """

        self.Rs = np.empty((self.number_of_variants, 3, 3))

        for i, row in enumerate(EULER_ANGLES):
            c1, c2, c3 = np.cos(np.deg2rad(row))
            s1, s2, s3 = np.sin(np.deg2rad(row))

            R = np.array(
                [
                    [c1 * c3 - c2 * s1 * s3, -c1 * s3 - c2 * c3 * s1, s1 * s2],
                    [c3 * s1 + c1 * c2 * s3, c1 * c2 * c3 - s1 * s3, -c1 * s2],
                    [s2 * s3, c3 * s2, c2],
                ]
            )
            R[np.where(abs(R) < 1e-8)] = 0.0
            self.Rs[i] = R

    def store_basal_slip_system(self):
        """
        BASAL SLIP SYSTEM
        """
        self.directionsSlipSystem[0, 0] = R2
        self.directionsSlipSystem[1, 0] = -R2 * R3
        self.directionsSlipSystem[2, 0] = ZERO

        self.directionsSlipSystem[0, 1] = R2
        self.directionsSlipSystem[1, 1] = R3 * R2
        self.directionsSlipSystem[2, 1] = ZERO

        self.directionsSlipSystem[0, 2] = -ONE
        self.directionsSlipSystem[1, 2] = ZERO
        self.directionsSlipSystem[2, 2] = ZERO

        self.normalsSlipSystem[0, :] = ZERO
        self.normalsSlipSystem[1, :] = ZERO
        self.normalsSlipSystem[2, :] = ONE

        self.IC += 3

    def store_prismatic_slip_system(self):
        """
        PRISMATIC SLIP SYSTEM
        """

        self.directionsSlipSystem[0, 0 + self.IC] = ONE
        self.directionsSlipSystem[1, 0 + self.IC] = ZERO
        self.directionsSlipSystem[2, 0 + self.IC] = ZERO

        self.directionsSlipSystem[0, 1 + self.IC] = R2
        self.directionsSlipSystem[1, 1 + self.IC] = R3 * R2
        self.directionsSlipSystem[2, 1 + self.IC] = ZERO

        self.directionsSlipSystem[0, 2 + self.IC] = -R2
        self.directionsSlipSystem[1, 2 + self.IC] = R3 * R2
        self.directionsSlipSystem[2, 2 + self.IC] = ZERO

        self.normalsSlipSystem[0, 0 + self.IC] = ZERO
        self.normalsSlipSystem[1, 0 + self.IC] = ONE
        self.normalsSlipSystem[2, 0 + self.IC] = ZERO

        self.normalsSlipSystem[0, 1 + self.IC] = -R3 * R2
        self.normalsSlipSystem[1, 1 + self.IC] = R2
        self.normalsSlipSystem[2, 1 + self.IC] = ZERO

        self.normalsSlipSystem[0, 2 + self.IC] = -R3 * R2
        self.normalsSlipSystem[1, 2 + self.IC] = -R2
        self.normalsSlipSystem[2, 2 + self.IC] = ZERO

        self.IC += 3

    def store_pyramidal_A_slip(self):
        """
        PYRAMIDAL A SLIP
        """
        BUNBO = np.sqrt(FOUR * (ratioCOverA) ** 2 + THREE)
        BUNSHI = ratioCOverA

        self.directionsSlipSystem[0, 0 + self.IC] = ONE
        self.directionsSlipSystem[1, 0 + self.IC] = ZERO
        self.directionsSlipSystem[2, 0 + self.IC] = ZERO

        self.directionsSlipSystem[0, 1 + self.IC] = R2
        self.directionsSlipSystem[1, 1 + self.IC] = R2 * R3
        self.directionsSlipSystem[2, 1 + self.IC] = ZERO

        self.directionsSlipSystem[0, 2 + self.IC] = -R2
        self.directionsSlipSystem[1, 2 + self.IC] = R2 * R3
        self.directionsSlipSystem[2, 2 + self.IC] = ZERO

        self.directionsSlipSystem[0, 3 + self.IC] = -ONE
        self.directionsSlipSystem[1, 3 + self.IC] = ZERO
        self.directionsSlipSystem[2, 3 + self.IC] = ZERO

        self.directionsSlipSystem[0, 4 + self.IC] = -R2
        self.directionsSlipSystem[1, 4 + self.IC] = -R2 * R3
        self.directionsSlipSystem[2, 4 + self.IC] = ZERO

        self.directionsSlipSystem[0, 5 + self.IC] = R2
        self.directionsSlipSystem[1, 5 + self.IC] = -R2 * R3
        self.directionsSlipSystem[2, 5 + self.IC] = ZERO

        self.normalsSlipSystem[0, 0 + self.IC] = ZERO
        self.normalsSlipSystem[1, 0 + self.IC] = -TWO * BUNSHI / BUNBO
        self.normalsSlipSystem[2, 0 + self.IC] = R3 / BUNBO

        self.normalsSlipSystem[0, 1 + self.IC] = R3 * BUNSHI / BUNBO
        self.normalsSlipSystem[1, 1 + self.IC] = -BUNSHI / BUNBO
        self.normalsSlipSystem[2, 1 + self.IC] = R3 / BUNBO

        self.normalsSlipSystem[0, 2 + self.IC] = R3 * BUNSHI / BUNBO
        self.normalsSlipSystem[1, 2 + self.IC] = BUNSHI / BUNBO
        self.normalsSlipSystem[2, 2 + self.IC] = R3 / BUNBO

        self.normalsSlipSystem[0, 3 + self.IC] = ZERO
        self.normalsSlipSystem[1, 3 + self.IC] = TWO * BUNSHI / BUNBO
        self.normalsSlipSystem[2, 3 + self.IC] = R3 / BUNBO

        self.normalsSlipSystem[0, 4 + self.IC] = -R3 * BUNSHI / BUNBO
        self.normalsSlipSystem[1, 4 + self.IC] = BUNSHI / BUNBO
        self.normalsSlipSystem[2, 4 + self.IC] = R3 / BUNBO

        self.normalsSlipSystem[0, 5 + self.IC] = -R3 * BUNSHI / BUNBO
        self.normalsSlipSystem[1, 5 + self.IC] = -BUNSHI / BUNBO
        self.normalsSlipSystem[2, 5 + self.IC] = R3 / BUNBO

        self.IC += 6

    def store_first_order_CpA_slip(self):
        """
        1ST ORDER <C+A> SLIP
        """

        BUNBO = np.sqrt(FOUR * (ratioCOverA) ** 2 + THREE)
        BUNSHI = ratioCOverA
        BUNB = TWO * (np.sqrt((ratioCOverA) ** 2 + ONE))

        self.directionsSlipSystem[0, 0 + self.IC] = ONE / BUNB
        self.directionsSlipSystem[1, 0 + self.IC] = R3 / BUNB
        self.directionsSlipSystem[2, 0 + self.IC] = TWO * BUSH / BUNB

        self.directionsSlipSystem[0, 1 + self.IC] = -ONE / BUNB
        self.directionsSlipSystem[1, 1 + self.IC] = R3 / BUNB
        self.directionsSlipSystem[2, 1 + self.IC] = TWO * BUSH / BUNB

        self.directionsSlipSystem[0, 2 + self.IC] = -TWO / BUNB
        self.directionsSlipSystem[1, 2 + self.IC] = ZERO
        self.directionsSlipSystem[2, 2 + self.IC] = TWO * BUSH / BUNB

        self.directionsSlipSystem[0, 3 + self.IC] = -ONE / BUNB
        self.directionsSlipSystem[1, 3 + self.IC] = -R3 / BUNB
        self.directionsSlipSystem[2, 3 + self.IC] = TWO * BUSH / BUNB

        self.directionsSlipSystem[0, 4 + self.IC] = ONE / BUNB
        self.directionsSlipSystem[1, 4 + self.IC] = -R3 / BUNB
        self.directionsSlipSystem[2, 4 + self.IC] = TWO * BUSH / BUNB

        self.directionsSlipSystem[0, 5 + self.IC] = TWO / BUNB
        self.directionsSlipSystem[1, 5 + self.IC] = ZERO
        self.directionsSlipSystem[2, 5 + self.IC] = TWO * BUSH / BUNB

        self.directionsSlipSystem[0, 6 + self.IC] = -ONE / BUNB
        self.directionsSlipSystem[1, 6 + self.IC] = R3 / BUNB
        self.directionsSlipSystem[2, 6 + self.IC] = TWO * BUSH / BUNB

        self.directionsSlipSystem[0, 7 + self.IC] = -TWO / BUNB
        self.directionsSlipSystem[1, 7 + self.IC] = ZERO
        self.directionsSlipSystem[2, 7 + self.IC] = TWO * BUSH / BUNB

        self.directionsSlipSystem[0, 8 + self.IC] = -ONE / BUNB
        self.directionsSlipSystem[1, 8 + self.IC] = -R3 / BUNB
        self.directionsSlipSystem[2, 8 + self.IC] = TWO * BUSH / BUNB

        self.directionsSlipSystem[0, 9 + self.IC] = ONE / BUNB
        self.directionsSlipSystem[1, 9 + self.IC] = -R3 / BUNB
        self.directionsSlipSystem[2, 9 + self.IC] = TWO * BUSH / BUNB

        self.directionsSlipSystem[0, 10 + self.IC] = TWO / BUNB
        self.directionsSlipSystem[1, 10 + self.IC] = ZERO
        self.directionsSlipSystem[2, 10 + self.IC] = TWO * BUSH / BUNB

        self.directionsSlipSystem[0, 11 + self.IC] = ONE / BUNB
        self.directionsSlipSystem[1, 11 + self.IC] = R3 / BUNB
        self.directionsSlipSystem[2, 11 + self.IC] = TWO * BUSH / BUNB

        self.normalsSlipSystem[0, 0 + self.IC] = ZERO
        self.normalsSlipSystem[1, 0 + self.IC] = -TWO * BUNSHI / BUNBO
        self.normalsSlipSystem[2, 0 + self.IC] = R3 / BUNBO

        self.normalsSlipSystem[0, 1 + self.IC] = R3 * BUNSHI / BUNBO
        self.normalsSlipSystem[1, 1 + self.IC] = -BUNSHI / BUNBO
        self.normalsSlipSystem[2, 1 + self.IC] = R3 / BUNBO

        self.normalsSlipSystem[0, 2 + self.IC] = R3 * BUNSHI / BUNBO
        self.normalsSlipSystem[1, 2 + self.IC] = BUNSHI / BUNBO
        self.normalsSlipSystem[2, 2 + self.IC] = R3 / BUNBO

        self.normalsSlipSystem[0, 3 + self.IC] = ZERO
        self.normalsSlipSystem[1, 3 + self.IC] = TWO * BUNSHI / BUNBO
        self.normalsSlipSystem[2, 3 + self.IC] = R3 / BUNBO

        self.normalsSlipSystem[0, 4 + self.IC] = -R3 * BUNSHI / BUNBO
        self.normalsSlipSystem[1, 4 + self.IC] = BUNSHI / BUNBO
        self.normalsSlipSystem[2, 4 + self.IC] = R3 / BUNBO

        self.normalsSlipSystem[0, 5 + self.IC] = -R3 * BUNSHI / BUNBO
        self.normalsSlipSystem[1, 5 + self.IC] = -BUNSHI / BUNBO
        self.normalsSlipSystem[2, 5 + self.IC] = R3 / BUNBO

        self.normalsSlipSystem[0, 6 + self.IC] = ZERO
        self.normalsSlipSystem[1, 6 + self.IC] = -TWO * BUNSHI / BUNBO
        self.normalsSlipSystem[2, 6 + self.IC] = R3 / BUNBO

        self.normalsSlipSystem[0, 7 + self.IC] = R3 * BUNSHI / BUNBO
        self.normalsSlipSystem[1, 7 + self.IC] = -BUNSHI / BUNBO
        self.normalsSlipSystem[2, 7 + self.IC] = R3 / BUNBO

        self.normalsSlipSystem[0, 8 + self.IC] = R3 * BUNSHI / BUNBO
        self.normalsSlipSystem[1, 8 + self.IC] = BUNSHI / BUNBO
        self.normalsSlipSystem[2, 8 + self.IC] = R3 / BUNBO

        self.normalsSlipSystem[0, 9 + self.IC] = ZERO
        self.normalsSlipSystem[1, 9 + self.IC] = TWO * BUNSHI / BUNBO
        self.normalsSlipSystem[2, 9 + self.IC] = R3 / BUNBO

        self.normalsSlipSystem[0, 10 + self.IC] = -R3 * BUNSHI / BUNBO
        self.normalsSlipSystem[1, 10 + self.IC] = BUNSHI / BUNBO
        self.normalsSlipSystem[2, 10 + self.IC] = R3 / BUNBO

        self.normalsSlipSystem[0, 11 + self.IC] = -R3 * BUNSHI / BUNBO
        self.normalsSlipSystem[1, 11 + self.IC] = -BUNSHI / BUNBO
        self.normalsSlipSystem[2, 11 + self.IC] = R3 / BUNBO

        self.IC += 12

    def store_second_order_CpA_slip(self):
        """
        2ND ORDER <C+A> SLIP
        """

        BUNB = TWO * (np.sqrt((ratioCOverA) ** 2 + ONE))

        self.directionsSlipSystem[0, 0 + self.IC] = -ONE / BUNB
        self.directionsSlipSystem[1, 0 + self.IC] = R3 / BUNB
        self.directionsSlipSystem[2, 0 + self.IC] = TWO * BUSH / BUNB

        self.directionsSlipSystem[0, 1 + self.IC] = -TWO / BUNB
        self.directionsSlipSystem[1, 1 + self.IC] = ZERO
        self.directionsSlipSystem[2, 1 + self.IC] = TWO * BUSH / BUNB

        self.directionsSlipSystem[0, 2 + self.IC] = -ONE / BUNB
        self.directionsSlipSystem[1, 2 + self.IC] = -R3 / BUNB
        self.directionsSlipSystem[2, 2 + self.IC] = TWO * BUSH / BUNB

        self.directionsSlipSystem[0, 3 + self.IC] = ONE / BUNB
        self.directionsSlipSystem[1, 3 + self.IC] = -R3 / BUNB
        self.directionsSlipSystem[2, 3 + self.IC] = TWO * BUSH / BUNB

        self.directionsSlipSystem[0, 4 + self.IC] = TWO / BUNB
        self.directionsSlipSystem[1, 4 + self.IC] = ZERO
        self.directionsSlipSystem[2, 4 + self.IC] = TWO * BUSH / BUNB

        self.directionsSlipSystem[0, 5 + self.IC] = ONE / BUNB
        self.directionsSlipSystem[1, 5 + self.IC] = R3 / BUNB
        self.directionsSlipSystem[2, 5 + self.IC] = TWO * BUSH / BUNB

        self.normalsSlipSystem[0, 0 + self.IC] = BUSH / BUNB
        self.normalsSlipSystem[1, 0 + self.IC] = -R3 * BUSH / BUNB
        self.normalsSlipSystem[2, 0 + self.IC] = TWO / BUNB

        self.normalsSlipSystem[0, 1 + self.IC] = TWO * BUSH / BUNB
        self.normalsSlipSystem[1, 1 + self.IC] = ZERO
        self.normalsSlipSystem[2, 1 + self.IC] = TWO / BUNB

        self.normalsSlipSystem[0, 2 + self.IC] = BUSH / BUNB
        self.normalsSlipSystem[1, 2 + self.IC] = R3 * BUSH / BUNB
        self.normalsSlipSystem[2, 2 + self.IC] = TWO / BUNB

        self.normalsSlipSystem[0, 3 + self.IC] = -BUSH / BUNB
        self.normalsSlipSystem[1, 3 + self.IC] = R3 * BUSH / BUNB
        self.normalsSlipSystem[2, 3 + self.IC] = TWO / BUNB

        self.normalsSlipSystem[0, 4 + self.IC] = -TWO * BUSH / BUNB
        self.normalsSlipSystem[1, 4 + self.IC] = ZERO
        self.normalsSlipSystem[2, 4 + self.IC] = TWO / BUNB

        self.normalsSlipSystem[0, 5 + self.IC] = -BUSH / BUNB
        self.normalsSlipSystem[1, 5 + self.IC] = -R3 * BUSH / BUNB
        self.normalsSlipSystem[2, 5 + self.IC] = TWO / BUNB

        self.IC = self.IC + 6

    def calculateSingleMPrime(self, iVariant, iSlipSystem, jVariant, jSlipSystem):
        """
        n1 --> n_in
        d1 --> d_in
        n2 --> n_out
        d2 --> d_out
        """
        R1 = self.Rs[iVariant]
        n1 = R1 @ self.normalsSlipSystem[:, iSlipSystem][:, None]
        d1 = R1 @ self.directionsSlipSystem[:, iSlipSystem][:, None]

        R2 = self.Rs[jVariant]
        n2 = R2 @ self.normalsSlipSystem[:, jSlipSystem][:, None]
        d2 = R2 @ self.directionsSlipSystem[:, jSlipSystem][:, None]

        m_prime = n1.T.dot(n2) * d1.T.dot(d2)
        return abs(m_prime[0, 0])

    def calculateMPrime(self):
        n = self.number_of_variants * self.number_of_slip_systems
        self.m_prime = np.zeros((n, n))

        for iVariant in range(self.number_of_variants):
            for iSlipSystem in range(self.number_of_slip_systems):
                for jVariant in range(iVariant, self.number_of_variants):
                    for jSlipSystem in range(self.number_of_slip_systems):
                        i = (iVariant * self.number_of_slip_systems) + iSlipSystem
                        j = (jVariant * self.number_of_slip_systems) + jSlipSystem
                        if i != j:
                            self.m_prime[i, j] = self.calculateSingleMPrime(
                                iVariant, iSlipSystem, jVariant, jSlipSystem
                            )
        self.m_prime += self.m_prime.T 

import numpy as np
import scipy.optimize as so


class RateConstantModel():
    def __init__(self, num_species=2, num_reactions=3, rates=[], alpha=0.0, lamb=1.0, method='SLSQP',
                 tol=1e-16, approx_jac=False, ODE_env=''):
        self.alpha = alpha
        self.lamb = lamb
        self.N = num_species
        self.R = num_reactions
        self.rates = rates
        self.init_xi = np.zeros_like(self.rates)
        #self.init_xi = self.rates
        self.method = method
        self.tol = tol
        self.approx_jac = approx_jac
        self.ODE_env = ODE_env

        # self.rates

    def compute_theta(self, Z, species_constant):
        zeros = np.zeros(len(Z[:, 0]))
        ones = np.ones(len(Z[:, 0]))

        '''
        # Lotka Volterra
        '''
        y1 = np.array(np.transpose([Z[:, 0], zeros]))   #y1 =[X,0]
        y2 = np.array(np.transpose([- Z[:, 0] * Z[:, 1], Z[:, 0] * Z[:, 1]])) #[-XY, XY]
        y3 =  np.array(np.transpose([zeros, -Z[:, 1]])) #[0, -Y]


        if species_constant != []:

            A = species_constant[0]
            B = species_constant[1]
            '''
            Brusselator
            '''
            y4 = np.array(np.transpose([A * ones, zeros])) #y4 = [A,0]
            y5 = np.array(np.transpose([Z[:, 0] ** 2 * Z[:, 1], -Z[:, 0] ** 2 * Z[:, 1]])) #y5 = [X^2*Y, -X^2*Y]
            y6 = np.array(np.transpose([- B * Z[:, 0], B * Z[:, 0]])) #y6 = [-BX, BX]
            y7 = np.array(np.transpose([-Z[:, 0], zeros])) #y7 = [-X,0]

        if self.ODE_env == "LV":
            theta = np.transpose([y1, y2, y3], (1, 0, 2))
        elif self.ODE_env == "Brusselator":
            theta = np.transpose([y4, y5, y6, y7], (1, 0, 2))
        elif self.ODE_env == "Generalized":
            theta = np.transpose([y1, y2, y3, y4, y5, y6], (1, 0, 2))

        elif self.ODE_env == "Oregonator":

            f = 1

            y8 = np.array(np.transpose([A * Z[:, 1], -A * Z[:, 1], zeros])) #y8 = [AY, -AY, 0]
            y9 = np.array(np.transpose([-Z[:, 0] * Z[:, 1], -Z[:, 0] * Z[:, 1], zeros])) #y9 = [-XY, -XY, 0]
            y10 = np.array(np.transpose([A * Z[:, 0], zeros, 2 * A * Z[:, 0]])) #y10 = [AX, 0, 2AX]
            y11 = np.array(np.transpose([-2 * Z[:, 0] ** 2, zeros, zeros])) #y11 = [-2X^2, 0, 0]
            y12 = np.array(np.transpose([zeros, 0.5 * f * B * Z[:, 2], - B * Z[:, 2]])) #y12 = [0, 1/2*f*BZ, -BZ]

            theta = np.transpose([y8, y9, y10, y11, y12], (1, 0, 2))
            #print("bruh",theta.shape)

        return theta

    def elastic_net_func(self, propensities, Z_arr, theta_arr, dt, alpha, lamb, u):

        num_species = self.N
        num_reactions = self.R
        result = 0
        total_time_steps = 0

        for i in range(len(theta_arr)):
            theta = theta_arr[i]
            time_steps = len(theta)
            Z = Z_arr[i] * int(1 / dt)
            dZ = np.gradient(Z)
            for t in range(1, time_steps - 1):  # for t in range(time_steps):
                for s in range(num_species):
                    x = dZ[0][t][s] - u[s]  # dZ
                    for r in range(num_reactions):
                        x -= propensities[r] * theta[t][r][s]

                    result += x ** 2

            total_time_steps += time_steps

        result *= 1.0 / (2.0 * total_time_steps)

        regulator = 0

        l1_regulator = 0
        for r in range(num_reactions):
            l1_regulator += abs(propensities[r])
        l1_regulator *= alpha * lamb
        regulator += l1_regulator

        if alpha != 0 and lamb < 1.0:
            l2_regulator = 0
            for r in range(num_reactions):
                l2_regulator += propensities[r] ** 2
            l2_regulator *= alpha * (1 - lamb)
            regulator += l2_regulator

        return result + regulator

    def elastic_net_jac(self, propensities, Z_arr, theta_arr, dt, alpha, lamb, u):

        num_species = self.N
        num_reactions = self.R

        result = np.zeros(num_reactions)
        total_time_steps = 0

        for i in range(len(theta_arr)):
            theta = theta_arr[i]
            time_steps = len(theta)
            Z = Z_arr[i] * int(1 / dt)
            dZ = np.gradient(Z)
            total_time_steps += time_steps
            for j in range(num_reactions):

                for t in range(time_steps):
                    for s in range(num_species):
                        x = dZ[0][t][s] - u[s]  # dZ
                        theta_t_j_s = theta[t][j][s]
                        for r in range(num_reactions):
                            x -= propensities[r] * theta[t][r][s]

                        result[j] += theta_t_j_s * x

        result /= -total_time_steps

        for j in range(num_reactions):
            # l1 regulator
            result[j] += alpha * lamb
            # l2 regulator
            result[j] += 2 * alpha * (1 - lamb) * propensities[j]

        return result

    def solve_minimize(self, Z_arr, theta_arr, dt, u):
        def objective(x):
            obj = self.elastic_net_func(x, Z_arr, theta_arr, dt, self.alpha, self.lamb, u)
            return obj

        jac = False if self.approx_jac else \
            lambda x: self.elastic_net_jac(x, Z_arr, theta_arr, dt, self.alpha, self.lamb, u)


        cons = ({'type': 'ineq',
                 'fun': lambda x: x[0]},
                {'type': 'ineq',
                 'fun': lambda x: x[1]},
                {'type': 'ineq',
                 'fun': lambda x: x[2]},
                {'type': 'ineq',
                 'fun': lambda x: x[3]},
                {'type': 'ineq',
                 'fun': lambda x: x[4]})

        if self.ODE_env == "Oregonator":
            constraints = cons
        else:
            constraints = None

        result = so.minimize(
            objective,
            x0=self.rates,
            bounds=None,
            tol=self.tol,
            method=self.method,
            jac=None,
            options=None,
            constraints = constraints)
        return result
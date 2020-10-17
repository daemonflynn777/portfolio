K = (3.0 - sqrt(5)) / 2.0
        x = a + K*(c - a)
        w = a + K*(c - a)
        v = a + K*(c - a)
        u = a + K*(c - a)
        #print(abs(K))
        u = c
        eps = (0.1)**flt_num
        tol = eps*abs(x) + eps*0.1
        #f_x = self.func(self.u_k + x*(self.u_k_line - self.u_k))
        #f_w = self.func(self.u_k + x*(self.u_k_line - self.u_k))
        #f_v = self.func(self.u_k + x*(self.u_k_line - self.u_k))
        d_cur = c - a
        d_prv = c - a
        iteration = 0
        #max_iter = 100
        flag = 0
        #while round(abs(x - (a + c)/2.0) + (c - a)/2.0, flt_num) > 2.0*tol:
        while iteration <= max_iter:
            f_x = self.func(self.u_k + x*(self.u_k_line - self.u_k))
            f_w = self.func(self.u_k + w*(self.u_k_line - self.u_k))
            f_v = self.func(self.u_k + v*(self.u_k_line - self.u_k))
            #print(iteration)
            print(a, c)
            #print(x)
            if round(max(x - a, c - x), flt_num) < eps or iteration == max_iter:
            #if abs(u - x) < eps or iteration == max_iter:    
                return x, self.func(self.u_k + x*(self.u_k_line - self.u_k))
            g = d_prv/2.0
            d_prv = d_cur
            np.seterr()
            if ((x != w and w != v and x != v) and (f_x != f_w and f_w != f_v and f_x != f_v)):
                u = x - (((x - v)**2)*(f_x - f_w) - ((x - w)**2)*(f_x - f_v))/(2*((x - v)*(f_x - f_w) - (x - w)*(f_x - f_v)))
            #print(u)
            #print(x)
            if u is None or u < a or u > c or abs(u - x) > g:
                if x < (a + c)/2.0:
                    u = x + K*(c - x)
                    d_prv = c - x
                else:
                    u = x - K*(x - a)
                    d_prv = x - a
            d_cur = abs(u - x)
            #f_x = self.func(self.u_k + x*(self.u_k_line - self.u_k))
            #f_w = self.func(self.u_k + x*(self.u_k_line - self.u_k))
            #f_v = self.func(self.u_k + x*(self.u_k_line - self.u_k))
            f_u = self.func(self.u_k + u*(self.u_k_line - self.u_k))
            if f_u > f_x:
                if u < x:
                    a = u
                else:
                    c = u
                if f_u <= f_w or w == x:
                    v = w
                    w = u
                else:
                    if f_u <= f_v or v == x or v == w:
                        v = u
            else:
                if u < x:
                    c = x
                else:
                    a = x
                v = w
                w = x
                x = u
            iteration += 1
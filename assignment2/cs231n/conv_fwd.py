













          out[n, f, out_y, out_x] =\
            np.sum(
              w[f,:,:,:] *\
              p[n, :, y:y+HH, x:x+WW] )
            + b[f]



















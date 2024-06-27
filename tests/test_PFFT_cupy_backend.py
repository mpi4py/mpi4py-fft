def test_PFFT_cupy_backend():
    import numpy as np
    import cupy as cp
    from mpi4py import MPI
    from mpi4py_fft import PFFT, newDistArray
    
    comm = MPI.COMM_WORLD
    
    # Set global size of the computational box
    N = np.array([comm.size * 8] * 3, dtype=int)
    expected_shape = (N[0] // comm.size, N[1], N[2])
    axes = ((0,), (1, 2))
    
    backends = ['numpy', 'cupy']
    FFTs = {backend: PFFT(comm, N, axes=axes, grid=(-1,), backend=backend) for backend in backends}
    assert FFTs['numpy'].axes == FFTs['cupy'].axes
    
    us = {backend: newDistArray(FFTs[backend], forward_output=False) for backend in backends}
    us['numpy'][:] = np.random.random(us['numpy'].shape).astype(us['numpy'].dtype)
    us['cupy'][:] = cp.array(us['numpy'])
    
    
    
    for backend, xp in zip(backends, [np, cp]):
        us['hat_' + backend] = newDistArray(FFTs[backend], forward_output=True)
        us['hat_' + backend] = FFTs[backend].forward(us[backend], us['hat_' + backend])
        us['back_and_forth_' + backend] = xp.zeros_like(us[backend])
        us['back_and_forth_' + backend] = FFTs[backend].backward(us['hat_' + backend], us['back_and_forth_' + backend])
    
        assert xp.allclose(us[backend], us['back_and_forth_' + backend]), f'Got different values after back and forth transformation with {backend} backend'
        assert np.allclose(us['back_and_forth_' + backend].shape, expected_shape), f"Got unexpected shape {us['back_and_forth_' + backend].shape} when expecting {expected_shape} with {backend} backend"
    assert np.allclose(us['hat_cupy'].get(), us['hat_numpy']), 'Got different values in frequency space'


if __name__ == '__main__':
    test_PFFT_cupy_backend()

HandLine detection(OpenCV Hand detection) & Skeleton Detection



Using opencv version 4.2 and CUDA version 11.2

Skeleton detection captures the characteristic points of the knuckles and marks them.

![image](https://user-images.githubusercontent.com/42111518/131345438-a43689d2-5cf3-4076-a0e5-b70c15d6b9b2.png)

![image](https://user-images.githubusercontent.com/42111518/131346527-0d76de5f-fe17-4102-9268-fdf24d640f24.png)


The base is Mask R CNN. Existing Mask R CNN displays areas, but skeleton detection only displays feature points excluding mask areas.

To use this program, pose_iter_102000.caffemodel and pose_deploy.prototxt are required, and please create a folder called hand and download it.

Skeleton detection is a 2-stack detection, so it takes a lot of delay. However, this program has been further optimized using the GPU to work with Hand Line detection.

It can also be operated in a Linux environment, so please match the path.
(Env : Windows , Linux)

![6666](https://user-images.githubusercontent.com/42111518/131348135-f9e1f050-709b-496b-887f-110648d170ff.png)

This is a double the efficiency frame improvement over conventional skeleton detection.

Always available when needed for camera-related vision tasks

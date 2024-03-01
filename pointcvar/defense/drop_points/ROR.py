"""ROR defense """
import torch
import torch.nn as nn
import open3d as o3d 
import numpy as np

class RORDefense(nn.Module):
    """Statistical outlier removal as defense.
    """

    def __init__(self, n=2, r=1.1):
        """SOR defense.
        Args:
            k (int, optional): kNN. Defaults to 2.
            alpha (float, optional): \miu + \alpha * std. Defaults to 1.1.
        """
        super(RORDefense, self).__init__()

        self.n = n
        self.r = r

    def outlier_removal(self, x):
        """Removes large kNN distance points.
        Args:
            x (torch.FloatTensor): batch input pc, [B, K, 3]
        Returns:
            torch.FloatTensor: pc after outlier removal, [B, N, 3]
        """
        pc = x.clone().detach().double()
        B, K = pc.shape[:2]
        new_pc = []
        for i in range(B):
          pcd = o3d.geometry.PointCloud()
          pcd.points = o3d.utility.Vector3dVector(pc[i].cpu().numpy())
          new_pc.append(pcd.remove_radius_outlier(nb_points=self.n, radius=self.r)[0])

        sel_pc = [RORDefense.pcd_to_torch(p).to(pc) for p in new_pc]
        return sel_pc

    @staticmethod
    def pcd_to_torch(x):
        return torch.from_numpy(np.array(x.points))

    def forward(self, x):
        with torch.no_grad():
            x = self.outlier_removal(x)
        return x

import numpy as np

from hymn.config import RANGE_STD


class ResidualMapCalculator:
    def __init__(self, tech, grid_resolution=0.5, grid_range_x=(0, 0), grid_range_y=(0, 0)):
        self.range_std = RANGE_STD[tech]
        self.grid_resolution = grid_resolution
        self.grid_range_x = grid_range_x
        self.grid_range_y = grid_range_y
        self._create_grid_points()

    def _create_grid_points(self):
        grid_x = np.arange(self.grid_range_x[0], self.grid_range_x[1], self.grid_resolution)
        grid_y = np.arange(self.grid_range_y[0], self.grid_range_y[1], self.grid_resolution)
        self.grid_x, self.grid_y = np.meshgrid(grid_x, grid_y, indexing='xy')
        self.grid_shape = self.grid_x.shape

    def calculate_residual_map(self, anchor_position, measurement):
        x_diff = anchor_position[0] - self.grid_x
        y_diff = anchor_position[1] - self.grid_y
        
        
        d = np.sqrt(x_diff**2 + y_diff**2)
        residuals = measurement - d
        std = self.range_std
            
        p = np.exp(-residuals**2 / (2 * std**2)) / np.sqrt(2 * np.pi * std**2)
        p[p < 1e-5] = 1e-5

        p /= np.sum(p)

        return p.astype(np.float32)
    
    def process_row(self, row, anchor_positions):
        measurement = np.array(row['ranges'], dtype = np.float32)
        
        residual_maps = np.zeros((len(anchor_positions), *self.grid_shape), dtype=np.float32)

        valid_mask = np.isfinite(measurement)
        valid_anchors = anchor_positions[valid_mask]
        valid_measurements = measurement[valid_mask]

        for i, (anchor_position, value) in enumerate(zip(valid_anchors, valid_measurements)):
            residual_maps[i] = self.calculate_residual_map(anchor_position, value)

        return residual_maps
    

def calculate(data, anchors, grid_res, grid_range_x, grid_range_y, tech):
    # Initialize the ResidualMapCalculator
    calculator = ResidualMapCalculator(tech, grid_resolution=grid_res, grid_range_x = grid_range_x, grid_range_y = grid_range_y)

    # Apply the process_row method to each row
    data['residual_range_map'] = data.apply(lambda row: calculator.process_row(row, anchor_positions= anchors), axis=1)

    return data
"""
Attachment controller for managing foot-to-region attachments in dynamic anchor simulation.
"""
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class SimulationConfig:
    """Configuration parameters for the attachment simulation."""
    region_radius: float = 0.1
    region_inactive_rgba: tuple = field(default_factory=lambda: (0.5, 0.5, 0.5, 0.25))
    region_active_rgba: tuple = field(default_factory=lambda: (0.2, 0.8, 0.2, 0.9))
    target_rtf: float = 1.0  # Real-time factor (1.0 = real-time)
    num_feet: int = 4
    num_regions: int = 4


class RobotAttachmentController:
    """
    Manages attachment state and logic for feet attaching to/detaching from regions.
    
    Tracks:
    - Which foot is attached to which region
    - Which region is occupied by which foot
    - Inhibition state for auto-attachment (prevents re-attachment until foot exits all regions)
    """

    def __init__(self, config: SimulationConfig = None):
        """
        Initialize the attachment controller.
        
        Args:
            config: SimulationConfig instance with parameters. Uses defaults if None.
        """
        self.config = config or SimulationConfig()
        
        # State: which region each foot is attached to (foot_key -> region_key or None)
        self.foot_attached_region: Dict[str, Optional[str]] = {
            str(i): None for i in range(1, self.config.num_feet + 1)
        }
        
        # State: which foot occupies each region (region_key -> foot_key or None)
        self.region_occupied_by: Dict[str, Optional[str]] = {
            str(i): None for i in range(1, self.config.num_regions + 1)
        }
        
        # Inhibit auto-reattach for feet that were manually detached
        self.foot_inhibit_autoattach: Dict[str, bool] = {
            str(i): False for i in range(1, self.config.num_feet + 1)
        }

    def get_foot_keys(self) -> list:
        """Get list of all foot keys."""
        return list(self.foot_attached_region.keys())

    def get_region_keys(self) -> list:
        """Get list of all region keys."""
        return list(self.region_occupied_by.keys())

    def is_foot_attached(self, foot_key: str) -> bool:
        """Check if a foot is currently attached to any region."""
        return self.foot_attached_region[foot_key] is not None

    def get_foot_region(self, foot_key: str) -> Optional[str]:
        """Get the region a foot is attached to, or None if free."""
        return self.foot_attached_region[foot_key]

    def is_region_occupied(self, region_key: str) -> bool:
        """Check if a region is currently occupied by a foot."""
        return self.region_occupied_by[region_key] is not None

    def get_region_occupant(self, region_key: str) -> Optional[str]:
        """Get the foot occupying a region, or None if free."""
        return self.region_occupied_by[region_key]

    def attach_foot_to_region(self, foot_key: str, region_key: str) -> bool:
        """
        Attach a foot to a region.
        
        Returns False if the foot or region is already occupied (no change made).
        Returns True if attachment succeeded.
        """
        # If this foot is already attached, don't re-attach
        if self.is_foot_attached(foot_key):
            return False
        
        # If region already has a foot, don't attach (one foot per region)
        if self.is_region_occupied(region_key):
            return False

        self.foot_attached_region[foot_key] = region_key
        self.region_occupied_by[region_key] = foot_key
        return True

    def detach_foot(self, foot_key: str) -> bool:
        """
        Detach a foot from its region and inhibit auto-reattachment.
        
        Returns False if the foot was not attached.
        Returns True if detachment succeeded.
        """
        region_key = self.foot_attached_region[foot_key]
        if region_key is None:
            return False

        self.foot_attached_region[foot_key] = None
        self.region_occupied_by[region_key] = None
        self.foot_inhibit_autoattach[foot_key] = True
        return True

    def check_foot_in_region(
        self,
        foot_pos: tuple,
        region_pos: tuple,
    ) -> bool:
        """
        Check if a foot position is within a region's radius.
        
        Args:
            foot_pos: (x, y, z) position of foot
            region_pos: (x, y, z) position of region center
            
        Returns:
            True if foot is within region_radius of region center
        """
        dx = foot_pos[0] - region_pos[0]
        dy = foot_pos[1] - region_pos[1]
        dz = foot_pos[2] - region_pos[2]
        dist_sq = dx * dx + dy * dy + dz * dz
        return dist_sq < self.config.region_radius * self.config.region_radius

    def update_inhibit_state(
        self,
        foot_key: str,
        foot_pos: tuple,
        region_positions: Dict[str, tuple],
    ) -> None:
        """
        Update inhibition state for a foot.
        
        If foot was inhibited and is now outside all regions, re-enable auto-attach.
        
        Args:
            foot_key: The foot to update
            foot_pos: Current foot position
            region_positions: Dict mapping region_key -> (x, y, z) position
        """
        if not self.foot_inhibit_autoattach[foot_key]:
            return

        # Check if foot is inside any region
        inside_any = False
        for region_key, region_pos in region_positions.items():
            if self.check_foot_in_region(foot_pos, region_pos):
                inside_any = True
                break

        if not inside_any:
            # Foot has exited all regions → allow auto-attach again
            self.foot_inhibit_autoattach[foot_key] = False

    def find_best_region(
        self,
        foot_pos: tuple,
        region_positions: Dict[str, tuple],
    ) -> Optional[str]:
        """
        Find the nearest free region within radius for a foot, or None if none available.
        
        Args:
            foot_pos: Current foot position
            region_positions: Dict mapping region_key -> (x, y, z) position
            
        Returns:
            Best available region key, or None
        """
        best_region = None
        best_dist_sq = None

        for region_key, region_pos in region_positions.items():
            # Skip occupied regions
            if self.is_region_occupied(region_key):
                continue

            if self.check_foot_in_region(foot_pos, region_pos):
                dx = foot_pos[0] - region_pos[0]
                dy = foot_pos[1] - region_pos[1]
                dz = foot_pos[2] - region_pos[2]
                dist_sq = dx * dx + dy * dy + dz * dz

                if best_region is None or dist_sq < best_dist_sq:
                    best_region = region_key
                    best_dist_sq = dist_sq

        return best_region

    def should_autoattach(self, foot_key: str) -> bool:
        """
        Check if a foot is eligible for auto-attachment.
        
        A foot can auto-attach if:
        - It's not already attached
        - It's not inhibited from auto-attaching
        
        Args:
            foot_key: The foot to check
            
        Returns:
            True if foot is eligible
        """
        return (
            not self.is_foot_attached(foot_key)
            and not self.foot_inhibit_autoattach[foot_key]
        )

    def reset(self) -> None:
        """Reset all attachment state to initial conditions (all free, no inhibition)."""
        for key in self.foot_attached_region:
            self.foot_attached_region[key] = None
        for key in self.region_occupied_by:
            self.region_occupied_by[key] = None
        for key in self.foot_inhibit_autoattach:
            self.foot_inhibit_autoattach[key] = False

class LevelControls:
    """
    Defines ways to interact with the environment during a level.
    Includes controls for navigating left and right, jumping, and
    restarting the level.
    """

       def __init__(self):
            self.left = "left"
            self.right = "right"
            self.jump = "jump"
            self.restart = "restart"

        def navigate_left(self):
            """Navigate the character left."""
            return self.left

        def navigate_right(self):
            """Navigate the character right."""
            return self.right

        def jump(self):
            """Jump the character."""
            return self.jump

        def restart(self):
            """Restart the level."""
            return self.restart

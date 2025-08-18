def convert_action(action):
    """Convert from the original format to the new format."""
    action_map = {
        "NOOP": "-",  # Nothing
        "Jump": "^",  # Jump
        "Right": ">",  # Right
        "Jump + Right": "/",  # Right Jump
        "Left": "<",  # Left
        "Jump + Left": "\\",  # Left Jump
    }
    return action_map.get(action, action)


def main():
    # Read the input file
    with open("actions/model_15/actions.txt", "r") as f:
        actions = f.read().strip().split(",")

    # Convert actions
    converted_actions = [convert_action(action.strip()) for action in actions]

    # Write to new file
    with open("actions/model_15/actions_converted.txt", "w") as f:
        f.write("".join(converted_actions))


if __name__ == "__main__":
    main()

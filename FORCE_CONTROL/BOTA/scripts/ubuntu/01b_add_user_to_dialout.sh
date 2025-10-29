# Exit on error
set -e

echo "Adding current user ($(whoami)) to the dialout group..."

# Add the current user to the dialout group
sudo usermod -a -G dialout $(whoami)

# Check if the user is already in the group or needs to log out and back in
if groups $(whoami) | grep -q '\bdialout\b'; then
    echo "User $(whoami) is already a member of the dialout group."
else
    echo "User $(whoami) has been added to the dialout group."
    echo "Important: You need to log out and log back in for this change to take effect."
fi

# Provide additional information
echo ""
echo "This grants access to serial devices without requiring sudo privileges."
echo "Common serial devices include:"
echo "  - /dev/ttyUSB*  (USB-to-serial adapters)"
echo "  - /dev/ttyACM*  (Arduino and similar devices)"
echo "  - /dev/ttyS*    (Hardware serial ports)"
echo ""

exit 0
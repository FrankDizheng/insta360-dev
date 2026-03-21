// Gemini 335 Camera Mount for NERO Robot Arm
// Installs into flange plate like gripper/teaching pendant
// Based on NERO manual section 2.3.2
// Units: mm

$fn = 60;

// ===== Flange interface (from manual) =====
// 4x M3x8 screws, tool inserts into flange center
flange_screw = 3.4;           // M3 clearance
flange_hole_spacing = 45.0;   // between hole groups
flange_insert_d = 30;         // insertion plug diameter (estimated)
flange_insert_h = 8;          // how deep the plug goes in
key_w = 16;                   // anti-fool key width
key_h = 8;                    // anti-fool key protrusion

// ===== Mounting plate =====
plate_w = 70;
plate_d = 28;
plate_t = 4;
plate_r = 5;

// ===== Camera (Gemini 335) =====
cam_w = 124;
cam_d = 27;
cam_h = 29;
cam_screw = 6.5;              // 1/4"-20 clearance

// ===== Arm =====
arm_h = 45;
arm_w = 28;
arm_t = 5;

// ===== Camera tilt =====
cam_tilt = 75;                // degrees from horizontal (75 = mostly looking down)

module rrect(w, d, h, r) {
    hull() for (x=[-1,1], y=[-1,1])
        translate([x*(w/2-r), y*(d/2-r), 0])
            cylinder(r=r, h=h);
}

// 1. Flange insertion plug — goes INTO the flange center hole
module flange_plug() {
    // Main cylindrical plug
    translate([0, 0, plate_t])
        cylinder(d=flange_insert_d, h=flange_insert_h);

    // Anti-fool key (oval protrusion for orientation)
    translate([-key_w/2, -flange_insert_d/2 - 1, plate_t])
        cube([key_w, key_h + 1, flange_insert_h]);
}

// 2. Mounting plate — sits flat against flange face, 4x M3 screw holes
module mount_plate() {
    difference() {
        rrect(plate_w, plate_d, plate_t, plate_r);

        // 4x M3 holes (2 groups of 2, spaced at 45mm)
        for (sx=[-1,1]) {
            translate([sx*flange_hole_spacing/2 - 4, -4, -1])
                cylinder(d=flange_screw, h=plate_t+2);
            translate([sx*flange_hole_spacing/2 + 4,  4, -1])
                cylinder(d=flange_screw, h=plate_t+2);
        }
    }
}

// 3. Arm — connects mounting plate to camera cradle
module arm() {
    // Main arm going down
    translate([-arm_w/2, -arm_t/2, -arm_h])
        cube([arm_w, arm_t, arm_h]);

    // Gussets for stiffness
    gusset = 12;
    for (sx=[-1,1])
        translate([sx*arm_w/2, 0, 0])
        rotate([90,0,0])
        translate([0, 0, -arm_t/2])
        linear_extrude(arm_t)
            polygon([[0,0], [sx*gusset,0], [0,-gusset]]);
}

// 4. Camera cradle — holds Gemini 335, tilted to look down
module camera_cradle() {
    wall = 3;
    cradle_w = cam_w + wall*2 + 2;
    cradle_d = cam_d + wall + 2;
    base_t = wall;
    lip = 5;

    translate([0, 0, -arm_h])
    rotate([cam_tilt, 0, 0])
    translate([0, 0, -base_t]) {
        difference() {
            union() {
                // Base plate (camera sits on this)
                translate([-cradle_w/2, -cradle_d/2, 0])
                    cube([cradle_w, cradle_d, base_t]);

                // Side walls
                for (sx=[-1,1])
                    translate([sx*(cradle_w/2 - wall), -cradle_d/2, 0])
                        cube([wall, cradle_d, cam_h*0.4]);

                // Back wall (taller, supports camera)
                translate([-cradle_w/2, cradle_d/2 - wall, 0])
                    cube([cradle_w, wall, cam_h*0.6]);

                // Front lip (low, doesn't block lens)
                translate([-cradle_w/2, -cradle_d/2, 0])
                    cube([cradle_w, wall, lip]);
            }

            // 1/4"-20 screw hole through base
            translate([0, 0, -1])
                cylinder(d=cam_screw, h=base_t+2);

            // Cable slot (back right)
            translate([cradle_w/4, cradle_d/2 - wall-1, base_t])
                cube([18, wall+2, cam_h*0.6]);
        }
    }
}

// 5. Cable channel along the arm
module cable_channel() {
    translate([-6, arm_t/2 - 2, -arm_h + 5])
        cube([12, 3, arm_h - 15]);
}

// ===== Assembly =====
color("SteelBlue") {
    difference() {
        union() {
            mount_plate();
            flange_plug();
            arm();
            camera_cradle();
        }
        cable_channel();
    }
}

// Ghost camera for size reference
%translate([0, 0, -arm_h])
    rotate([cam_tilt, 0, 0])
    translate([-cam_w/2, -cam_d/2, 0])
        cube([cam_w, cam_d, cam_h]);

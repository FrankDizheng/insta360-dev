// NERO End-Effector Flange Bracket
// A single 5mm thick plate with a 2D profile extruded
// Units: mm

$fn = 80;

plate_t = 5.0;
clamp_gap = 12.0;
mount_hole = 4.2;
clamp_hole = 3.4;
hole_spacing = 45.0;

// The entire bracket is one 2D profile extruded to 5mm thickness.
// Profile (front view, XZ plane):
//   - Top: bone-shaped ear plate, ~88mm wide
//   - Middle: narrow vertical neck, ~46mm wide
//   - Bottom: half-circle arc opening downward, ~50mm diameter
//   - Right side: small ear tab with 2 holes

module bracket_profile() {
    ear_w = 88;
    ear_h = 12;
    ear_r = 10;
    neck_w = 46;
    neck_h = 25;
    arc_r = 25;
    ear_tab_w = 12;
    ear_tab_h = 20;

    // Top ear plate (bone shape)
    hull() {
        translate([-(ear_w/2 - ear_r), neck_h + ear_h - ear_r])
            circle(r=ear_r);
        translate([(ear_w/2 - ear_r), neck_h + ear_h - ear_r])
            circle(r=ear_r);
        translate([-ear_w/2 + ear_r, neck_h])
            square([ear_w - ear_r*2, 1]);
    }

    // Neck (vertical section)
    translate([-neck_w/2, -5])
        square([neck_w, neck_h + 6]);

    // Arc section: thick-walled half ring opening downward
    arc_wall = (neck_w - clamp_gap) / 2;
    difference() {
        translate([0, -5])
            circle(r=arc_r + arc_wall);
        circle(r=arc_r);
        // Keep only bottom half
        translate([-(arc_r + arc_wall + 1), 0])
            square([(arc_r + arc_wall) * 2 + 2, arc_r + arc_wall + 1]);
        // Cut bottom gap
        translate([-clamp_gap/2, -(arc_r + arc_wall + 1)])
            square([clamp_gap, arc_r + arc_wall + 1]);
    }

    // Right-side ear tab
    translate([neck_w/2, -ear_tab_h + 5])
        square([ear_tab_w, ear_tab_h]);
}

module bracket_holes() {
    ear_h_center = 25 + 6;

    // 4x top mounting holes (2 groups of 2)
    for (sx=[-1,1]) {
        translate([sx*hole_spacing/2 - 4, ear_h_center - 3])
            circle(d=mount_hole);
        translate([sx*hole_spacing/2 + 4, ear_h_center + 3])
            circle(d=mount_hole);
    }

    // 2x clamp face holes
    translate([-8, -2]) circle(d=clamp_hole);
    translate([8, -2]) circle(d=clamp_hole);

    // 2x right ear holes
    translate([46/2 + 6, -3]) circle(d=clamp_hole);
    translate([46/2 + 6, -11]) circle(d=clamp_hole);

    // Cable routing slot
    translate([-6, 8])
        square([12, 8]);
}

// Extrude the profile to plate thickness
color("DimGray")
translate([0, 0, -plate_t/2])
linear_extrude(height=plate_t)
    difference() {
        bracket_profile();
        bracket_holes();
    }

//! Per-camera on-sensor PDAF (phase-detect autofocus) pixel row patterns.
//!
//! On-sensor PDAF pixels replace ordinary green photosites on specific,
//! camera-model-specific rows, and read back brighter than their neighbors -
//! left uncorrected, demosaic spreads them into small colored specks. There
//! is no way to know which rows are affected without knowing the sensor's
//! PDAF layout, so unlike the rest of `raw_preprocess.rs` this needs a real
//! per-camera-model data table rather than a general statistical rule.
//!
//! Ported from ART's `rtengine/camconst.json` (the `pdaf_pattern`/
//! `pdaf_offset` fields) - RapidRAW is AGPLv3 and ART/RawTherapee is GPLv3,
//! which is compatible (GPLv3 content may be combined into an AGPLv3 work).
//! Only the PDAF-relevant subset is ported here, not camconst.json's much
//! larger (and for us irrelevant - rawler already handles this) black/white
//! level and color-matrix data. Source:
//! <https://github.com/artraweditor/ART/blob/master/rtengine/camconst.json>.
//!
//! `pattern` is a list of row offsets (relative to `offset`) that repeats
//! every `pattern.last() + step` rows for the full sensor height - see
//! `pdaf_rows_near`.

pub struct PdafPattern {
    pub cameras: &'static [&'static str],
    pub pattern: &'static [i32],
    pub offset: i32,
}

pub const PDAF_PATTERNS: &[PdafPattern] = &[
    PdafPattern {
        cameras: &[
            "FUJIFILM GFX 100",
            "FUJIFILM GFX100S",
            "FUJIFILM GFX 100S",
            "FUJIFILM GFX 100 II",
        ],
        pattern: &[0, 18],
        offset: 0,
    },
    PdafPattern {
        cameras: &["Nikon Z 7"],
        pattern: &[0, 12],
        offset: 29,
    },
    PdafPattern {
        cameras: &["Nikon Z 6"],
        pattern: &[0, 12],
        offset: 32,
    },
    PdafPattern {
        // Every 12th line, blue subpixel rows (lower black-frame stddev).
        cameras: &["Nikon Z 50"],
        pattern: &[
            285, 297, 309, 321, 333, 345, 357, 369, 381, 393, 405, 417, 429, 441, 453, 465, 477,
            489, 501, 513, 525, 537, 549, 561, 573, 585, 597, 609, 621, 633, 645, 657, 669, 681,
            693, 705, 717, 729, 741, 753, 765, 777, 789, 801, 813, 825, 837, 849, 861, 873, 885,
            897, 909, 921, 933, 945, 957, 969, 981, 993, 1005, 1017, 1029, 1041, 1053, 1065, 1077,
            1089, 1101, 1113, 1125, 1137, 1149, 1161, 1173, 1185, 1197, 1209, 1221, 1233, 1245,
            1257, 1269, 1281, 1293, 1305, 1317, 1329, 1341, 1353, 1365, 1377, 1389, 1401, 1413,
            1425, 1437, 1449, 1461, 1473, 1485, 1497, 1509, 1521, 1533, 1545, 1557, 1569, 1581,
            1593, 1605, 1617, 1629, 1641, 1653, 1665, 1677, 1689, 1701, 1713, 1725, 1737, 1749,
            1761, 1773, 1785, 1797, 1809, 1821, 1833, 1845, 1857, 1869, 1881, 1893, 1905, 1917,
            1929, 1941, 1953, 1965, 1977, 1989, 2001, 2013, 2025, 2037, 2049, 2061, 2073, 2085,
            2097, 2109, 2121, 2133, 2145, 2157, 2169, 2181, 2193, 2205, 2217, 2229, 2241, 2253,
            2265, 2277, 2289, 2301, 2313, 2325, 2337, 2349, 2361, 2373, 2385, 2397, 2409, 2421,
            2433, 2445, 2457, 2469, 2481, 2493, 2505, 2517, 2529, 2541, 2553, 2565, 2577, 2589,
            2601, 2613, 2625, 2637, 2649, 2661, 2673, 2685, 2697, 2709, 2721, 2733, 2745, 2757,
            2769, 2781, 2793, 2805, 2817, 2829, 2841, 2853, 2865, 2877, 2889, 2901, 2913, 2925,
            2937, 2949, 2961, 2973, 2985, 2997, 3009, 3021, 3033, 3045, 3057, 3069, 3081, 3093,
            3105, 3117, 3129, 3141, 3153, 3165, 3177, 3189, 3201, 3213, 3225, 3237, 3249, 3261,
            3273, 3285, 3297, 3309, 3321, 3333, 3345, 3357, 3369, 3381, 3393, 3405, 3417, 3429,
            3441,
        ],
        offset: 0,
    },
    PdafPattern {
        cameras: &["Sony ILCE-6000"],
        pattern: &[
            0, 12, 36, 54, 72, 90, 114, 126, 144, 162, 180, 204, 216, 240, 252, 270, 294, 306, 324,
            342, 366, 384, 396, 414, 432, 450, 474, 492, 504, 522, 540, 564, 576, 594, 606, 630,
        ],
        offset: 3,
    },
    PdafPattern {
        cameras: &["Sony ILCE-6100", "Sony ILCE-6400", "Sony ILCE-6600"],
        pattern: &[
            0, 12, 36, 54, 72, 90, 114, 126, 144, 162, 180, 204, 216, 240, 252, 270, 294, 306, 324,
            342, 366, 384, 396, 414, 432, 450, 474, 492, 504, 522, 540, 564, 576, 594, 606, 630,
        ],
        offset: 3,
    },
    PdafPattern {
        cameras: &["Sony ILCE-6300", "Sony ILCE-6500"],
        pattern: &[
            0, 12, 36, 54, 72, 90, 114, 126, 144, 162, 180, 204, 216, 240, 252, 270, 294, 306, 324,
            342, 366, 384, 396, 414, 432, 450, 474, 492, 504, 522, 540, 564, 576, 594, 606, 630,
        ],
        offset: 3,
    },
    PdafPattern {
        cameras: &["Sony ILCE-7RM2", "Sony DSC-RX1RM2"],
        pattern: &[
            0, 24, 36, 60, 84, 120, 132, 156, 192, 204, 240, 252, 276, 300, 324, 360, 372, 396, 420,
        ],
        offset: 31,
    },
    PdafPattern {
        cameras: &["Sony ILCE-7M3"],
        pattern: &[
            0, 12, 24, 36, 54, 66, 72, 84, 96, 114, 120, 132, 150, 156, 174, 180, 192, 204, 216,
            234, 240, 252, 264, 276, 282, 300, 306, 324, 336, 342, 360, 372, 384, 402, 414, 420,
        ],
        offset: 9,
    },
    PdafPattern {
        cameras: &["Sony ILCE-7RM3"],
        pattern: &[
            0, 24, 36, 60, 84, 120, 132, 156, 192, 204, 240, 252, 276, 300, 324, 360, 372, 396,
            420, 444, 480, 492, 504, 540, 564, 576, 612, 636, 660, 696, 720, 732, 756, 780, 804,
            840,
        ],
        offset: 31,
    },
    PdafPattern {
        cameras: &["Sony ILCE-9", "Sony ILCE-9M2"],
        pattern: &[
            0, 12, 24, 36, 54, 66, 72, 84, 96, 114, 120, 132, 150, 156, 174, 180, 192, 204, 216,
            234, 240, 252, 264, 276, 282, 300, 306, 324, 336, 342, 360, 372, 384, 402, 414, 420,
        ],
        offset: -7,
    },
    PdafPattern {
        // 7RM5 assumed to share the 7CR's sensor. Per ART's own comment,
        // this pattern is a composite reconstructed from a bug report and
        // may repeat with a longer true period than shown here.
        cameras: &["Sony ILCE-7CR", "Sony ILCE-7RM5"],
        pattern: &[
            0, 12, 18, 36, 42, 60, 66, 72, 78, 96, 108, 120, 126, 138, 156, 168, 180, 186, 192,
            198, 210, 222, 228, 240, 246, 252, 270, 276, 282, 288, 306, 312, 318, 330, 336, 348,
            360, 366, 372, 378, 390, 396, 408, 420,
        ],
        offset: 1,
    },
    PdafPattern {
        cameras: &["Sony ZV-1"],
        pattern: &[0, 24, 48, 72, 88, 120],
        offset: 17,
    },
];

/// Looks up the PDAF pattern for a "<make> <model>" camera name
/// (case-insensitive, exact match against the known list).
pub fn lookup(camera_name: &str) -> Option<&'static PdafPattern> {
    let name = camera_name.trim();
    PDAF_PATTERNS
        .iter()
        .find(|p| p.cameras.iter().any(|c| c.eq_ignore_ascii_case(name)))
}

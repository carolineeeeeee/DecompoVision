import type { BBox, ImageInfo, TaskImageInfo } from "./types";
import {
  randomIntFromInterval,
  fillLargestBBox,
  fillLargestTaskBBox,
} from "./util";

let _imageUrls: Array<string> = [
  "https://detectionexp.s3.ca-central-1.amazonaws.com/JPEGImages/2011_002818.jpg",
  // "https://detectionexp.s3.ca-central-1.amazonaws.com/JPEGImages/2012_003651.jpg",
  // "https://detectionexp.s3.ca-central-1.amazonaws.com/JPEGImages/1377_2010_005853.jpg",
  // "https://detectionexp.s3.ca-central-1.amazonaws.com/JPEGImages/2011_003242.jpg",
  // "https://detectionexp.s3.ca-central-1.amazonaws.com/JPEGImages/1242_2008_006082.jpg",
  // "https://detectionexp.s3.ca-central-1.amazonaws.com/JPEGImages/2011_001173.jpg",
  // "https://detectionexp.s3.ca-central-1.amazonaws.com/JPEGImages/1840_2011_006819.jpg",
  // 'https://detectionexp.s3.ca-central-1.amazonaws.com/JPEGImages/1802_2009_002571.jpg',
  // 'https://detectionexp.s3.ca-central-1.amazonaws.com/JPEGImages/803_2008_004339.jpg',
  // 'https://detectionexp.s3.ca-central-1.amazonaws.com/JPEGImages/2008_006562.jpg',
  // 'https://detectionexp.s3.ca-central-1.amazonaws.com/JPEGImages/1827_2008_000703.jpg',
  // 'https://detectionexp.s3.ca-central-1.amazonaws.com/JPEGImages/2008_006178.jpg',
  // 'https://detectionexp.s3.ca-central-1.amazonaws.com/JPEGImages/653_2010_000433.jpg',
  // 'https://detectionexp.s3.ca-central-1.amazonaws.com/JPEGImages/444_2008_005408.jpg',
  // 'https://detectionexp.s3.ca-central-1.amazonaws.com/JPEGImages/1587_2010_005978.jpg',
  // 'https://detectionexp.s3.ca-central-1.amazonaws.com/JPEGImages/2010_004627.jpg',
  // 'https://detectionexp.s3.ca-central-1.amazonaws.com/JPEGImages/1387_2011_001928.jpg',
  // 'https://detectionexp.s3.ca-central-1.amazonaws.com/JPEGImages/2008_000851.jpg',
  // 'https://detectionexp.s3.ca-central-1.amazonaws.com/JPEGImages/2008_003334.jpg',
  // 'https://detectionexp.s3.ca-central-1.amazonaws.com/JPEGImages/2012_001425.jpg',
  // 'https://detectionexp.s3.ca-central-1.amazonaws.com/JPEGImages/266_2011_001891.jpg',
  // 'https://detectionexp.s3.ca-central-1.amazonaws.com/JPEGImages/834_2007_005430.jpg',
  // 'https://detectionexp.s3.ca-central-1.amazonaws.com/JPEGImages/355_2009_002073.jpg',
  // 'https://detectionexp.s3.ca-central-1.amazonaws.com/JPEGImages/2010_004059.jpg',
  // 'https://detectionexp.s3.ca-central-1.amazonaws.com/JPEGImages/2008_001267.jpg',
  // 'https://detectionexp.s3.ca-central-1.amazonaws.com/JPEGImages/2010_006364.jpg',
  // 'https://detectionexp.s3.ca-central-1.amazonaws.com/JPEGImages/243_2007_008142.jpg',
  // 'https://detectionexp.s3.ca-central-1.amazonaws.com/JPEGImages/2011_002818.jpg',
  // 'https://detectionexp.s3.ca-central-1.amazonaws.com/JPEGImages/2011_007180.jpg',
  // 'https://detectionexp.s3.ca-central-1.amazonaws.com/JPEGImages/2011_001173.jpg',
  // 'https://detectionexp.s3.ca-central-1.amazonaws.com/JPEGImages/2007_005304.jpg',
  // 'https://detectionexp.s3.ca-central-1.amazonaws.com/JPEGImages/2010_000283.jpg',
  // 'https://detectionexp.s3.ca-central-1.amazonaws.com/JPEGImages/2007_003106.jpg',
  // 'https://detectionexp.s3.ca-central-1.amazonaws.com/JPEGImages/2011_004733.jpg',
  // 'https://detectionexp.s3.ca-central-1.amazonaws.com/JPEGImages/2007_002470.jpg',
  // 'https://detectionexp.s3.ca-central-1.amazonaws.com/JPEGImages/2008_004778.jpg',
  // 'https://detectionexp.s3.ca-central-1.amazonaws.com/JPEGImages/246_2011_006385.jpg',
  // 'https://detectionexp.s3.ca-central-1.amazonaws.com/JPEGImages/2011_003016.jpg',
  // 'https://detectionexp.s3.ca-central-1.amazonaws.com/JPEGImages/1417_2008_002013.jpg',
  // 'https://detectionexp.s3.ca-central-1.amazonaws.com/JPEGImages/57_2012_001013.jpg',
  // 'https://detectionexp.s3.ca-central-1.amazonaws.com/JPEGImages/2011_006900.jpg',
  // 'https://detectionexp.s3.ca-central-1.amazonaws.com/JPEGImages/2010_001451.jpg',
  // 'https://detectionexp.s3.ca-central-1.amazonaws.com/JPEGImages/2011_002578.jpg',
  // 'https://detectionexp.s3.ca-central-1.amazonaws.com/JPEGImages/760_2010_001754.jpg',
  // 'https://detectionexp.s3.ca-central-1.amazonaws.com/JPEGImages/1178_2012_000027.jpg',
  // 'https://detectionexp.s3.ca-central-1.amazonaws.com/JPEGImages/1362_2008_005115.jpg',
  // 'https://detectionexp.s3.ca-central-1.amazonaws.com/JPEGImages/1564_2008_000489.jpg',
  // 'https://detectionexp.s3.ca-central-1.amazonaws.com/JPEGImages/2011_007132.jpg',
  // 'https://detectionexp.s3.ca-central-1.amazonaws.com/JPEGImages/1422_2009_002407.jpg',
  // 'https://detectionexp.s3.ca-central-1.amazonaws.com/JPEGImages/1850_2008_002679.jpg',
  // 'https://detectionexp.s3.ca-central-1.amazonaws.com/JPEGImages/2011_003591.jpg',
  // 'https://detectionexp.s3.ca-central-1.amazonaws.com/JPEGImages/2011_003016.jpg',
  // 'https://detectionexp.s3.ca-central-1.amazonaws.com/JPEGImages/2012_001926.jpg',
  // 'https://detectionexp.s3.ca-central-1.amazonaws.com/JPEGImages/2012_003176.jpg',
  // 'https://detectionexp.s3.ca-central-1.amazonaws.com/JPEGImages/1850_2008_002679.jpg',
  // 'https://detectionexp.s3.ca-central-1.amazonaws.com/JPEGImages/2011_003016.jpg',
  // 'https://detectionexp.s3.ca-central-1.amazonaws.com/JPEGImages/2012_001752.jpg',
  // 'https://detectionexp.s3.ca-central-1.amazonaws.com/JPEGImages/2008_004545.jpg',
  // 'https://detectionexp.s3.ca-central-1.amazonaws.com/JPEGImages/2010_000283.jpg',
  // 'https://detectionexp.s3.ca-central-1.amazonaws.com/JPEGImages/1729_2009_001906.jpg',
  // 'https://detectionexp.s3.ca-central-1.amazonaws.com/JPEGImages/1237_2011_006691.jpg',
  // 'https://detectionexp.s3.ca-central-1.amazonaws.com/JPEGImages/1249_2008_006227.jpg',
  // 'https://detectionexp.s3.ca-central-1.amazonaws.com/JPEGImages/1675_2008_000274.jpg',
  // 'https://detectionexp.s3.ca-central-1.amazonaws.com/JPEGImages/349_2010_003146.jpg',
  // 'https://detectionexp.s3.ca-central-1.amazonaws.com/JPEGImages/2007_008407.jpg',
  // 'https://detectionexp.s3.ca-central-1.amazonaws.com/JPEGImages/1998_2012_001171.jpg',
  // 'https://detectionexp.s3.ca-central-1.amazonaws.com/JPEGImages/1792_2008_000273.jpg',
  // 'https://detectionexp.s3.ca-central-1.amazonaws.com/JPEGImages/1278_2011_001116.jpg',
  // 'https://detectionexp.s3.ca-central-1.amazonaws.com/JPEGImages/2009_000097.jpg',
  // 'https://detectionexp.s3.ca-central-1.amazonaws.com/JPEGImages/2012_000849.jpg',
  // 'https://detectionexp.s3.ca-central-1.amazonaws.com/JPEGImages/2007_008407.jpg',
  // 'https://detectionexp.s3.ca-central-1.amazonaws.com/JPEGImages/504_2011_001655.jpg',
  // 'https://detectionexp.s3.ca-central-1.amazonaws.com/JPEGImages/1377_2010_005853.jpg',
  // 'https://detectionexp.s3.ca-central-1.amazonaws.com/JPEGImages/2010_003640.jpg',
  // 'https://detectionexp.s3.ca-central-1.amazonaws.com/JPEGImages/2011_000024.jpg',
  // 'https://detectionexp.s3.ca-central-1.amazonaws.com/JPEGImages/937_2011_005731.jpg',
  // 'https://detectionexp.s3.ca-central-1.amazonaws.com/JPEGImages/2008_001464.jpg',
  // 'https://detectionexp.s3.ca-central-1.amazonaws.com/JPEGImages/2008_008242.jpg',
  // 'https://detectionexp.s3.ca-central-1.amazonaws.com/JPEGImages/1827_2008_000703.jpg',
  // 'https://detectionexp.s3.ca-central-1.amazonaws.com/JPEGImages/213_2011_001392.jpg',
  // 'https://detectionexp.s3.ca-central-1.amazonaws.com/JPEGImages/1321_2011_004866.jpg',
  // 'https://detectionexp.s3.ca-central-1.amazonaws.com/JPEGImages/266_2011_001891.jpg',
  // 'https://detectionexp.s3.ca-central-1.amazonaws.com/JPEGImages/266_2011_001891.jpg',
  // 'https://detectionexp.s3.ca-central-1.amazonaws.com/JPEGImages/2011_006842.jpg',
  // 'https://detectionexp.s3.ca-central-1.amazonaws.com/JPEGImages/1850_2008_002679.jpg',
  // 'https://detectionexp.s3.ca-central-1.amazonaws.com/JPEGImages/962_2010_001514.jpg',
  // 'https://detectionexp.s3.ca-central-1.amazonaws.com/JPEGImages/1864_2008_000254.jpg',
  // 'https://detectionexp.s3.ca-central-1.amazonaws.com/JPEGImages/1792_2008_000273.jpg',
  // 'https://detectionexp.s3.ca-central-1.amazonaws.com/JPEGImages/2008_005146.jpg',
  // 'https://detectionexp.s3.ca-central-1.amazonaws.com/JPEGImages/2008_000703.jpg',
  // 'https://detectionexp.s3.ca-central-1.amazonaws.com/JPEGImages/2011_003016.jpg',
  // 'https://detectionexp.s3.ca-central-1.amazonaws.com/JPEGImages/2007_000032.jpg',
  // 'https://detectionexp.s3.ca-central-1.amazonaws.com/JPEGImages/1545_2010_000361.jpg',
  // 'https://detectionexp.s3.ca-central-1.amazonaws.com/JPEGImages/297_2010_005243.jpg',
  // 'https://detectionexp.s3.ca-central-1.amazonaws.com/JPEGImages/721_2008_008554.jpg',
  // 'https://detectionexp.s3.ca-central-1.amazonaws.com/JPEGImages/1802_2009_002571.jpg',
  // 'https://detectionexp.s3.ca-central-1.amazonaws.com/JPEGImages/2011_003016.jpg',
  // 'https://detectionexp.s3.ca-central-1.amazonaws.com/JPEGImages/2007_008407.jpg',
  // 'https://detectionexp.s3.ca-central-1.amazonaws.com/JPEGImages/1802_2009_002571.jpg',
  // 'https://detectionexp.s3.ca-central-1.amazonaws.com/JPEGImages/440_2008_001461.jpg',
  // 'https://detectionexp.s3.ca-central-1.amazonaws.com/JPEGImages/1982_2007_007531.jpg',
  // 'https://detectionexp.s3.ca-central-1.amazonaws.com/JPEGImages/2011_001001.jpg',
  // 'https://detectionexp.s3.ca-central-1.amazonaws.com/JPEGImages/1586_2010_002130.jpg',
];

let _bboxes: BBox[][] = [
  [[62, 2, 79, 176]],
  // [[13.75, 125.0, 63.125, 158.75]],
  // [[296.0, 168.0, 40.0, 212.0]],
  // [[72.0, 176.0, 321.0, 199.0]],
  // [[209, 440, 78, 39]],
  // [[2.0, 61.0, 476.0, 297.0]],
  // [[21, 57, 289, 437]],
  // [[273.125, 33.125, 225.625, 173.75]],
  // [[1.0, 95.0, 227.0, 83.0]],
  // [[33, 25, 336, 190]],
  // [[63, 250, 282, 212]],
  // [[68.85, 17.0, 194.65, 394.4]],
  // [[31.0, 32.0, 226.0, 292.0]],
  // [[90.0, 70.0, 371.0, 283.0]],
  // [[2.0, 128.0, 468.0, 154.0]],
  // [[131, 321, 54, 66]],
  // [[29.375, 71.25, 158.75, 205.625]],
  // [[35.0, 279.0, 40.0, 53.0]],
  // [[64.5, 41.0, 58.0, 130.5]],
  // [[96.0, 70.0, 404.0, 331.0]],
  // [[101, 137, 192, 127]],
  // [[23, 390, 300, 49]],
  // [[177, 14, 91, 469]],
  // [[57.5, 126.875, 376.875, 190.0]],
  // [[1.0, 1.0, 451.0, 324.0]],
  // [[219.375, 123.125, 70.625, 159.375]],
  // [[78, 193, 144, 285]],
  // [[38, 2, 269, 461]],
  // [[56.0, 5.0, 402.0, 327.0]],
  // [[37, 6, 335, 490]],
  // [[70, 14, 160, 101]],
  // [[7, 225, 186, 205]],
  // [[32, 199, 120, 283]],
  // [[209.0, 85.0, 177.0, 160.0]],
  // [[23.0, 85.0, 390.0, 375.0]],
  // [[1.0, 41.0, 207.0, 359.0]],
  // [[42.0, 4.0, 322.0, 484.0]],
  // [[97, 73, 71, 268]],
  // [[40.0, 1.0, 385.0, 330.0]],
  // [[66.25, 128.125, 210.625, 104.375]],
  // [[108, 135, 218, 161]],
  // [[37.0, 1.0, 277.0, 383.0]],
  // [[10.0, 1.0, 381.0, 372.0]],
  // [[205, 178, 93, 158]],
  // [[24, 16, 305, 414]],
  // [[20.0, 32.0, 289.0, 457.0]],
  // [[93.125, 50.625, 245.0, 228.75]],
  // [[8, 20, 262, 439]],
  // [[30, 67, 264, 312]],
  // [[16, 22, 218, 176]],
  // [[89.0, 67.0, 93.0, 152.0]],
  // [[112, 4, 173, 127]],
  // [[27, 24, 225, 427]],
  // [[148, 64, 106, 426]],
  // [[94.0, 388.0, 47.0, 73.0]],
  // [[11.875, 46.25, 440.625, 187.5]],
  // [[279.0, 34.0, 90.0, 206.0]],
  // [[203.0, 234.0, 35.0, 77.0]],
  // [[142.0, 72.0, 334.0, 260.0]],
  // [[1.0, 1.0, 499.0, 374.0]],
  // [[305.0, 8.0, 194.0, 366.0]],
  // [[125.0, 68.0, 65.0, 38.0]],
  // [[110, 44, 147, 311]],
  // [[33.0, 54.99999999999999, 179.51999999999998, 170.72]],
  // [[60.0, 165.625, 151.25, 98.75]],
  // [[370.625, 136.875, 10.0, 31.875]],
  // [[1.0, 1.0, 463.0, 109.0]],
  // [[165.625, 91.25, 292.5, 261.25]],
  // [[411.0, 152.0, 56.0, 27.0]],
  // [[111.875, 27.5, 326.875, 333.75]],
  // [[73.0, 175.0, 130.0, 106.0]],
  // [[376.875, 148.125, 16.875, 36.25]],
  // [[326, 185, 144, 24]],
  // [[41.0, 83.0, 10.0, 41.0]],
  // [[345.625, 208.125, 21.25, 100.625]],
  // [[165, 51, 140, 417]],
  // [[153.0, 192.0, 180.0, 198.0]],
  // [[29, 67, 456, 246]],
  // [[12.5, 101.875, 72.5, 75.0]],
  // [[68.0, 62.0, 84.0, 232.0]],
  // [[126.875, 221.875, 23.75, 33.125]],
  // [[352.0, 71.0, 16.0, 39.0]],
  // [[178, 326, 84, 91]],
  // [[134.375, 113.75, 198.125, 172.5]],
  // [[96.0, 98.0, 153.0, 213.0]],
  // [[-30.625, -30.625, 307.5, 168.75]],
  // [[3, 0, 471, 336]],
  // [[49, 250, 260, 32]],
  // [[159.0, 5.0, 168.0, 370.0]],
  // [[265.625, 90.625, 72.5, 80.625]],
  // [[171.25, 94.375, 56.875, 34.375]],
  // [[195.0, 180.0, 18.0, 49.0]],
  // [[353, 9, 18, 466]],
  // [[175.0, 241.0, 147.0, 92.0]],
  // [[-12.5, 285.0, 149.375, 80.625]],
  // [[293.0, 37.0, 179.0, 161.0]],
  // [[168.125, 117.5, 221.875, 153.75]],
  // [[12, 261, 356, 123]],
  // [[352.0, 1.0, 126.0, 204.0]],
  // [[196.0, 178.0, 80.0, 131.0]],
  // [[238.75, 39.375, 54.375, 108.125]],
  // [[252, 1, 30, 427]],
  // [[121.0, 157.0, 35.0, 36.0]],
];

export const sanityCheckData: ImageInfo[] = [
  {
    filename: "2008_000568.jpg",
    objects: [
      {
        bbox: [1, 40, 293, 335],
        name: "person",
      },
      {
        bbox: [360, 50, 140, 325],
        name: "person",
      },
    ],
    url: "https://mturk-host.s3.us-east-2.amazonaws.com/SanityCheckImages/2008_000568.jpg",
  },
  {
    filename: "2011_003106.jpg",
    objects: [
      {
        bbox: [69, 36, 269, 248],
        name: "person",
      },
    ],
    url: "https://mturk-host.s3.us-east-2.amazonaws.com/SanityCheckImages/2011_003106.jpg",
  },
  {
    filename: "2008_000634.jpg",
    objects: [
      {
        bbox: [157, 110, 60, 197],
        name: "person",
      },
      {
        bbox: [176, 126, 33, 71],
        name: "person",
      },
    ],
    url: "https://mturk-host.s3.us-east-2.amazonaws.com/SanityCheckImages/2008_000634.jpg",
  },
  {
    filename: "2008_000740.jpg",
    objects: [
      {
        bbox: [40, 10, 238, 445],
        name: "person",
      },
    ],
    url: "https://mturk-host.s3.us-east-2.amazonaws.com/SanityCheckImages/2008_000740.jpg",
  },
  {
    filename: "2008_000630.jpg",
    objects: [
      {
        bbox: [92, 155, 82, 256],
        name: "person",
      },
    ],
    url: "https://mturk-host.s3.us-east-2.amazonaws.com/SanityCheckImages/2008_000630.jpg",
  },
  {
    filename: "2011_002432.jpg",
    objects: [
      {
        bbox: [1, 87, 298, 413],
        name: "person",
      },
    ],
    url: "https://mturk-host.s3.us-east-2.amazonaws.com/SanityCheckImages/2011_002432.jpg",
  },
  {
    filename: "2008_000745.jpg",
    objects: [
      {
        bbox: [70, 16, 216, 463],
        name: "person",
      },
    ],
    url: "https://mturk-host.s3.us-east-2.amazonaws.com/SanityCheckImages/2008_000745.jpg",
  },
  {
    filename: "2011_002286.jpg",
    objects: [
      {
        bbox: [131, 196, 137, 253],
        name: "person",
      },
      {
        bbox: [43, 52, 106, 263],
        name: "person",
      },
    ],
    url: "https://mturk-host.s3.us-east-2.amazonaws.com/SanityCheckImages/2011_002286.jpg",
  },
  {
    filename: "2008_000535.jpg",
    objects: [
      {
        bbox: [221, 16, 81, 258],
        name: "person",
      },
      {
        bbox: [90, 89, 89, 207],
        name: "person",
      },
      {
        bbox: [157, 50, 63, 227],
        name: "person",
      },
    ],
    url: "https://mturk-host.s3.us-east-2.amazonaws.com/SanityCheckImages/2008_000535.jpg",
  },
  {
    filename: "2011_002309.jpg",
    objects: [
      {
        bbox: [78, 151, 129, 237],
        name: "person",
      },
    ],
    url: "https://mturk-host.s3.us-east-2.amazonaws.com/SanityCheckImages/2011_002309.jpg",
  },
  {
    filename: "2008_000700.jpg",
    objects: [
      {
        bbox: [22, 70, 358, 230],
        name: "person",
      },
    ],
    url: "https://mturk-host.s3.us-east-2.amazonaws.com/SanityCheckImages/2008_000700.jpg",
  },
  {
    filename: "2008_000475.jpg",
    objects: [
      {
        bbox: [23, 78, 134, 389],
        name: "person",
      },
      {
        bbox: [117, 82, 104, 271],
        name: "person",
      },
      {
        bbox: [188, 120, 103, 289],
        name: "person",
      },
    ],
    url: "https://mturk-host.s3.us-east-2.amazonaws.com/SanityCheckImages/2008_000475.jpg",
  },
  {
    filename: "2011_002663.jpg",
    objects: [
      {
        bbox: [182, 128, 90, 247],
        name: "person",
      },
      {
        bbox: [193, 99, 154, 276],
        name: "person",
      },
    ],
    url: "https://mturk-host.s3.us-east-2.amazonaws.com/SanityCheckImages/2011_002663.jpg",
  },
  {
    filename: "2008_000510.jpg",
    objects: [
      {
        bbox: [114, 46, 205, 239],
        name: "person",
      },
    ],
    url: "https://mturk-host.s3.us-east-2.amazonaws.com/SanityCheckImages/2008_000510.jpg",
  },
  {
    filename: "2011_003037.jpg",
    objects: [
      {
        bbox: [45, 28, 161, 416],
        name: "person",
      },
    ],
    url: "https://mturk-host.s3.us-east-2.amazonaws.com/SanityCheckImages/2011_003037.jpg",
  },
  {
    filename: "2011_002603.jpg",
    objects: [
      {
        bbox: [33, 137, 166, 238],
        name: "person",
      },
    ],
    url: "https://mturk-host.s3.us-east-2.amazonaws.com/SanityCheckImages/2011_002603.jpg",
  },
  {
    filename: "2011_002946.jpg",
    objects: [
      {
        bbox: [160, 53, 293, 322],
        name: "person",
      },
    ],
    url: "https://mturk-host.s3.us-east-2.amazonaws.com/SanityCheckImages/2011_002946.jpg",
  },
  {
    filename: "2008_000407.jpg",
    objects: [
      {
        bbox: [54, 151, 277, 349],
        name: "person",
      },
      {
        bbox: [75, 74, 211, 333],
        name: "person",
      },
    ],
    url: "https://mturk-host.s3.us-east-2.amazonaws.com/SanityCheckImages/2008_000407.jpg",
  },
  {
    filename: "2011_003136.jpg",
    objects: [
      {
        bbox: [98, 157, 277, 250],
        name: "person",
      },
    ],
    url: "https://mturk-host.s3.us-east-2.amazonaws.com/SanityCheckImages/2011_003136.jpg",
  },
  {
    filename: "2011_003082.jpg",
    objects: [
      {
        bbox: [58, 34, 253, 341],
        name: "person",
      },
    ],
    url: "https://mturk-host.s3.us-east-2.amazonaws.com/SanityCheckImages/2011_003082.jpg",
  },
];

export const qualificationTestData: ImageInfo[] = [
  {
    filename: "2008_001448.jpg",
    objects: [
      {
        bbox: [332, 126, 137, 43],
        name: "aeroplane",
      },
      {
        bbox: [1, 89, 297, 80],
        name: "aeroplane",
      },
    ],
    url: "https://mturk-host.s3.us-east-2.amazonaws.com/QualitativeTestImages/2008_001448.jpg",
  },
  {
    filename: "2007_002281.jpg",
    objects: [
      {
        bbox: [1, 30, 386, 183],
        name: "car",
      },
      {
        bbox: [189, 54, 25, 42],
        name: "person",
      },
    ],
    url: "https://mturk-host.s3.us-east-2.amazonaws.com/QualitativeTestImages/2007_002281.jpg",
  },
  {
    filename: "2011_001061.jpg",
    objects: [
      {
        bbox: [122, 76, 86, 212],
        name: "person",
      },
    ],
    url: "https://mturk-host.s3.us-east-2.amazonaws.com/QualitativeTestImages/2011_001061.jpg",
  },
  {
    filename: "2008_001843.jpg",
    objects: [
      {
        bbox: [340, 208, 148, 73],
        name: "boat",
      },
      {
        bbox: [14, 204, 195, 95],
        name: "boat",
      },
      {
        bbox: [225, 215, 41, 14],
        name: "boat",
      },
    ],
    url: "https://mturk-host.s3.us-east-2.amazonaws.com/QualitativeTestImages/2008_001843.jpg",
  },
  {
    filename: "2007_000042.jpg",
    objects: [
      {
        bbox: [263, 32, 237, 263],
        name: "train",
      },
      {
        bbox: [1, 36, 234, 263],
        name: "train",
      },
    ],
    url: "https://mturk-host.s3.us-east-2.amazonaws.com/QualitativeTestImages/2007_000042.jpg",
  },
  {
    filename: "2011_000959.jpg",
    objects: [
      {
        bbox: [34, 70, 406, 305],
        name: "person",
      },
    ],
    url: "https://mturk-host.s3.us-east-2.amazonaws.com/QualitativeTestImages/2011_000959.jpg",
  },
  {
    filename: "2011_001177.jpg",
    objects: [
      {
        bbox: [261, 126, 90, 234],
        name: "person",
      },
    ],
    url: "https://mturk-host.s3.us-east-2.amazonaws.com/QualitativeTestImages/2011_001177.jpg",
  },
  {
    filename: "2008_002042.jpg",
    objects: [
      {
        bbox: [91, 133, 45, 117],
        name: "person",
      },
      {
        bbox: [148, 127, 63, 201],
        name: "person",
      },
      {
        bbox: [287, 119, 37, 144],
        name: "person",
      },
      {
        bbox: [24, 121, 10, 18],
        name: "person",
      },
    ],
    url: "https://mturk-host.s3.us-east-2.amazonaws.com/QualitativeTestImages/2008_002042.jpg",
  },
  {
    filename: "2011_000928.jpg",
    objects: [
      {
        bbox: [93, 203, 125, 172],
        name: "person",
      },
    ],
    url: "https://mturk-host.s3.us-east-2.amazonaws.com/QualitativeTestImages/2011_000928.jpg",
  },
  {
    filename: "2008_000723.jpg",
    objects: [
      {
        bbox: [91, 92, 162, 282],
        name: "person",
      },
      {
        bbox: [228, 92, 147, 282],
        name: "person",
      },
    ],
    url: "https://mturk-host.s3.us-east-2.amazonaws.com/QualitativeTestImages/2008_000723.jpg",
  },
  {
    filename: "2008_001820.jpg",
    objects: [
      {
        bbox: [253, 45, 247, 276],
        name: "bus",
      },
      {
        bbox: [1, 18, 255, 311],
        name: "bus",
      },
    ],
    url: "https://mturk-host.s3.us-east-2.amazonaws.com/QualitativeTestImages/2008_001820.jpg",
  },
  {
    filename: "2007_002227.jpg",
    objects: [
      {
        bbox: [419, 64, 81, 71],
        name: "tvmonitor",
      },
      {
        bbox: [33, 20, 417, 310],
        name: "bicycle",
      },
    ],
    url: "https://mturk-host.s3.us-east-2.amazonaws.com/QualitativeTestImages/2007_002227.jpg",
  },
  {
    filename: "2011_001038.jpg",
    objects: [
      {
        bbox: [114, 111, 184, 178],
        name: "person",
      },
    ],
    url: "https://mturk-host.s3.us-east-2.amazonaws.com/QualitativeTestImages/2011_001038.jpg",
  },
  {
    filename: "2011_000907.jpg",
    objects: [
      {
        bbox: [183, 82, 108, 172],
        name: "person",
      },
    ],
    url: "https://mturk-host.s3.us-east-2.amazonaws.com/QualitativeTestImages/2011_000907.jpg",
  },
  {
    filename: "2011_001143.jpg",
    objects: [
      {
        bbox: [251, 64, 210, 311],
        name: "person",
      },
      {
        bbox: [21, 51, 191, 324],
        name: "person",
      },
    ],
    url: "https://mturk-host.s3.us-east-2.amazonaws.com/QualitativeTestImages/2011_001143.jpg",
  },
  {
    filename: "2011_001222.jpg",
    objects: [
      {
        bbox: [179, 87, 291, 285],
        name: "person",
      },
    ],
    url: "https://mturk-host.s3.us-east-2.amazonaws.com/QualitativeTestImages/2011_001222.jpg",
  },
  {
    filename: "2007_003051.jpg",
    objects: [
      {
        bbox: [30, 82, 253, 125],
        name: "car",
      },
    ],
    url: "https://mturk-host.s3.us-east-2.amazonaws.com/QualitativeTestImages/2007_003051.jpg",
  },
  {
    filename: "2007_002760.jpg",
    objects: [
      {
        bbox: [256, 28, 244, 249],
        name: "cat",
      },
      {
        bbox: [1, 7, 272, 287],
        name: "cat",
      },
    ],
    url: "https://mturk-host.s3.us-east-2.amazonaws.com/QualitativeTestImages/2007_002760.jpg",
  },
  {
    filename: "2007_000925.jpg",
    objects: [
      {
        bbox: [29, 101, 187, 242],
        name: "sheep",
      },
      {
        bbox: [309, 105, 151, 228],
        name: "sheep",
      },
    ],
    url: "https://mturk-host.s3.us-east-2.amazonaws.com/QualitativeTestImages/2007_000925.jpg",
  },
  {
    filename: "2011_001243.jpg",
    objects: [
      {
        bbox: [1, 76, 355, 424],
        name: "person",
      },
    ],
    url: "https://mturk-host.s3.us-east-2.amazonaws.com/QualitativeTestImages/2011_001243.jpg",
  },
];

fillLargestBBox(sanityCheckData);
fillLargestBBox(qualificationTestData);

export const noiseUrl =
  "https://mturk-host.s3.us-east-2.amazonaws.com/classification/noise.jpeg";

export const countdownTime = 4000;
export const totalWaitingTimePerImage = 20000;

const urlEles: HTMLCollectionOf<Element> =
  document.getElementsByClassName("urls");
if (urlEles) {
  const urls = Array.from(urlEles).map((ele) => ele.getAttribute("url"));
  if (urls.length && urls[0] && urls[0].includes("https")) {
    _imageUrls = urls.filter((url) => url !== null) as string[];
  }
}
const bboxEles: HTMLCollectionOf<Element> =
  document.getElementsByClassName("bboxes");

if (bboxEles && bboxEles.length) {
  const firstBbox = bboxEles[0].getAttribute("bbox");
  if (!(firstBbox && firstBbox.includes("$"))) {
    const bboxes: BBox[][] = Array.from(bboxEles).map((ele) => {
      const bbox = ele.getAttribute("bbox");
      return bbox ? JSON.parse(bbox) : bbox;
    });
    if (bboxes && bboxes[0]) {
      _bboxes = bboxes;
    }
  }
}

// insert 2 random image urls and bboxes
// pick 2 sanity check images
const images = [];
let whileLoopCount = 0; // used to prevent infinite loop
const sanityCheckUrlSet = new Set<string>();
while (images.length < 2) {
  if (whileLoopCount > 100) {
    throw new Error("Could not find 2 sanity check images");
  }
  const randomImg =
    sanityCheckData[randomIntFromInterval(0, sanityCheckData.length - 1)];
  if (!sanityCheckUrlSet.has(randomImg.url)) {
    images.push(randomImg);
  }
  whileLoopCount++;
}

// const sanityCheckIndicesSet = new Set<number>([]);
// while (sanityCheckIndicesSet.size < 2) {
//   sanityCheckIndicesSet.add(randomIntFromInterval(0, images.length - 1));
// }
// const sanityCheckIndices = Array.from(sanityCheckIndicesSet).sort();
images.forEach((image) => {
  const insertIdx = randomIntFromInterval(0, _imageUrls.length - 1);
  _imageUrls.splice(insertIdx, 0, image.url);
  _bboxes.splice(
    insertIdx,
    0,
    image.objects.map((obj) => obj.bbox)
  );
});

// export image urls and bboxes
// !image urls and bboxes must match (in terms of order)
const imageUrls = [..._imageUrls];
const bboxes = [..._bboxes];
export const sanityCheckImages = images;
export const taskImageData: TaskImageInfo[] = imageUrls.map((url, idx) => ({
  url,
  objects: bboxes[idx].map((bbox) => ({ bbox })),
}));
fillLargestTaskBBox(taskImageData);
export const noiseTime = 1000;
export const bboxTime = 400;
export const imageTime = 300;
export const classes: string[] = [
  "person",
  "bird",
  "cat",
  "cow",
  "dog",
  "horse",
  "sheep",
  "aeroplane",
  "bicycle",
  "boat",
  "bus",
  "car",
  "motorbike",
  "train",
  "bottle",
  "chair",
  "dining table",
  "potted plant",
  "sofa",
  "tv/monitor",
  "other",
];
classes.sort();

export const qualificationCode = "1024";

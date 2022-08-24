import type { ImageInfo } from "./types";

import { randomIntFromInterval, fillLargestBBox } from "./util";

let _imageUrls: Array<string> = [
  // "https://i.imgur.com/MkYwSlh.jpg",
  // "https://i.imgur.com/1mntCy4.jpg"
  // "https://i.imgur.com/AqOsCVn.jpg",
  // "https://i.imgur.com/1kDAmjn.jpg",
  // "https://i.imgur.com/ibX6R0k.jpg",
  // "https://i.imgur.com/OXA6HL0.jpg",
  // "https://i.imgur.com/iyBUbak.jpg",
  // "https://i.imgur.com/Yz79vnt.jpg",
  // "https://i.imgur.com/WaqFF6f.jpg",
  // "https://i.imgur.com/G3v5O3n.jpg",
  // "https://i.imgur.com/EAmFFqq.jpg",
  // "https://i.imgur.com/k54kvKM.jpg",
  // "https://i.imgur.com/xKPxcoK.jpg",
  // "https://i.imgur.com/e2UbXnK.jpg",
  // "https://i.imgur.com/U1B4FYM.jpg",
  // 'https://detectionexp.s3.ca-central-1.amazonaws.com/JPEGImages/2008_003624.jpg',
  // 'https://detectionexp.s3.ca-central-1.amazonaws.com/JPEGImages/852_2010_000630.jpg',
  // 'https://detectionexp.s3.ca-central-1.amazonaws.com/JPEGImages/2012_000857.jpg',
  // 'https://detectionexp.s3.ca-central-1.amazonaws.com/JPEGImages/474_2009_002366.jpg',
  // 'https://detectionexp.s3.ca-central-1.amazonaws.com/JPEGImages/101_2010_001074.jpg',
  // 'https://detectionexp.s3.ca-central-1.amazonaws.com/JPEGImages/1257_2009_002066.jpg',
  // 'https://detectionexp.s3.ca-central-1.amazonaws.com/JPEGImages/2011_005168.jpg',
  // 'https://detectionexp.s3.ca-central-1.amazonaws.com/JPEGImages/269_2008_000400.jpg',
  // 'https://detectionexp.s3.ca-central-1.amazonaws.com/JPEGImages/2008_001267.jpg',
  // 'https://detectionexp.s3.ca-central-1.amazonaws.com/JPEGImages/1802_2009_002571.jpg',
  // 'https://detectionexp.s3.ca-central-1.amazonaws.com/JPEGImages/1152_2011_002034.jpg',
  // 'https://detectionexp.s3.ca-central-1.amazonaws.com/JPEGImages/2008_002013.jpg',
  // 'https://detectionexp.s3.ca-central-1.amazonaws.com/JPEGImages/2010_003999.jpg',
  // 'https://detectionexp.s3.ca-central-1.amazonaws.com/JPEGImages/1193_2011_007188.jpg',
  // 'https://detectionexp.s3.ca-central-1.amazonaws.com/JPEGImages/2009_001241.jpg',
  // 'https://detectionexp.s3.ca-central-1.amazonaws.com/JPEGImages/1907_2010_001317.jpg',
  // 'https://detectionexp.s3.ca-central-1.amazonaws.com/JPEGImages/1313_2010_005758.jpg',
  // 'https://detectionexp.s3.ca-central-1.amazonaws.com/JPEGImages/694_2010_004877.jpg',
  // 'https://detectionexp.s3.ca-central-1.amazonaws.com/JPEGImages/1599_2012_000656.jpg',
  "https://detectionexp.s3.ca-central-1.amazonaws.com/JPEGImages/692_2011_003424.jpg",
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
];

fillLargestBBox(sanityCheckData);
fillLargestBBox(qualificationTestData);

export const countdownTime = 4000;
export const totalWaitingTimePerImage = 20000;

// let _bboxTimeouts: number[] = _imageUrls.map(() => countdownTime);
// let _bboxImageTimeouts: number[] = _imageUrls.map(
//   () => totalWaitingTimePerImage
// );
const urlEles: HTMLCollectionOf<Element> =
  document.getElementsByClassName("urls");
// const bboxTimeoutEles: HTMLCollectionOf<Element> =
//   document.getElementsByClassName("bbox-timeout");
// const bboxImageTimeoutEles: HTMLCollectionOf<Element> =
//   document.getElementsByClassName("image-timeout");

// if (bboxTimeoutEles && urlEles && bboxTimeoutEles.length !== urlEles.length) {
//   console.warn(
//     "Number of url elements doesn't match with number of timeout elements"
//   );
// }

if (urlEles && urlEles.length) {
  const urls: string[] = Array.from(urlEles).map((ele) =>
    ele.getAttribute("url")
  ) as string[];
  console.log(urls);
  if (urls.length && urls[0] && urls[0].includes("https")) {
    _imageUrls = urls;
    console.log("loaded imageUrls from csv");
    console.log(_imageUrls);
  }
}

// if (bboxTimeoutEles.length) {
//   const delays = Array.from(bboxTimeoutEles).map((ele) =>
//     ele.getAttribute("time")
//   );
//   if (!delays.every((delay) => delay !== null)) {
//     throw new Error("Unexpected Delay Value, there should not be any null");
//   }
//   // the following line converts string | null to number by setting null to 10000, which should never occur
//   // I had to deal with this. The previous block has already checked for null, so this should not be a problem
//   const delaysNum = delays.map((delay) => (delay ? parseInt(delay) : 10000));
//   _bboxTimeouts = delaysNum;
//   console.log("loaded bboxTimeouts from csv");
//   console.log(_bboxTimeouts);
// }

// if (bboxImageTimeoutEles.length) {
//   const delays_ = Array.from(bboxImageTimeoutEles).map((ele) =>
//     ele.getAttribute("time")
//   );
//   if (!delays_.every((delay) => delay !== null)) {
//     throw new Error("Unexpected Delay Value, there should not be any null");
//   }
//   // the following line converts string | null to number by setting null to 10000, which should never occur
//   // I had to deal with this. The previous block has already checked for null, so this should not be a problem
//   const delaysNum = delays_.map((delay) => (delay ? parseInt(delay) : 10000));
//   _bboxImageTimeouts = delaysNum;
//   console.log("loaded bboxImageTimeouts from csv");
//   console.log(_bboxImageTimeouts);
// }

// insert sanity check data
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

// console.log(images);
for (let i = 0; i < images.length; i++) {
  const randIdx = randomIntFromInterval(0, _imageUrls.length - 1);
  _imageUrls.splice(randIdx, 0, images[i].url);
  // _bboxTimeouts.splice(randIdx, 0, countdownTime);
  // _bboxImageTimeouts.splice(randIdx, 0, totalWaitingTimePerImage);
}

export const imageUrls = [..._imageUrls];
export const bboxTimeouts = imageUrls.map(() => countdownTime);
export const bboxImageTimeouts = imageUrls.map(() => totalWaitingTimePerImage);
// export const bboxTimeouts = [..._bboxTimeouts];
// export const bboxImageTimeouts = [..._bboxImageTimeouts];

export const qualificationCode = "1020";
export const goodIouThreshold = 0.5;
export const qualificationTestPassThreshold = 0.5; // percentage of images that need to be passed

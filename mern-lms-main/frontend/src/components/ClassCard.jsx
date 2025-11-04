import { useEffect, useState } from "react";
import { Link } from "react-router-dom";

export default function ClassCard({ classId, isList }) {
  const [classInfo, setClassInfo] = useState([]);
  const [classImage, setClassImage] = useState("");
  const imageURLs = [
    "https://img.freepik.com/free-psd/realistic-school-supplies_23-2150588345.jpg",
    "https://img.freepik.com/free-vector/geometric-science-education-background-vector-gradient-blue-digital-remix_53876-125993.jpg",
    "https://img.freepik.com/free-vector/education-pattern-background-doodle-style_53876-115365.jpg",
    "https://img.freepik.com/free-vector/gradient-international-day-education-background_23-2151120677.jpg",
    "https://img.freepik.com/free-vector/hand-drawn-back-school-background_23-2149464866.jpg",
    "https://img.freepik.com/premium-photo/back-school-equipment-premium-psd_467500-32.jpg",
    "https://img.freepik.com/free-photo/desk-workspace-with-various-elements_23-2148043273.jpg",
    "https://img.freepik.com/free-photo/elevated-view-laptop-stationeries-blue-backdrop_23-2147880457.jpg",
    "https://img.freepik.com/free-photo/flat-lay-arrangement-desk-elements-with-copy-space_23-2148513316.jpg",
    "https://img.freepik.com/free-photo/blue-surface-with-study-tools_23-2147864592.jpg",
  ];

  useEffect(() => {
    const stringToDecimal = (str) => {
      // Handle empty or non-string input
      if (!str || typeof str !== "string") {
        return 1;
      }

      // Convert string to a number based on character codes
      let sum = 0;
      for (let i = 0; i < str.length; i++) {
        sum += str.charCodeAt(i);
      }

      // Normalize to a positive integer (1-1000 range)
      const normalized = sum % 1000 || 1;

      // Ensure it's a positive integer
      return Math.max(1, Math.floor(normalized));
    };
    const fetchClassInfo = async () => {
      try {
        if (classId) {
          const res = await fetch(`/api/class/get-info/${classId}`);
          const data = await res.json();
          if (res.ok) {
            setClassInfo(data);
            const uniqueString =
              data?.classItem?.name +
              data?.subject?.name +
              data?.subject?.code +
              data?.course?.startAcademicYear +
              data?.course?.endAcademicYear;
            const index = stringToDecimal(uniqueString) % imageURLs.length;
            setClassImage(imageURLs[index]);
          }
        }
      } catch (error) {
        console.log(error.message);
      }
    };
    fetchClassInfo();
  }, [classId]);

  if (!isList) {
    return (
      <div className="w-full border-2 border-gray-300 rounded-xl overflow-hidden shadow hover:shadow-xl transition-shadow duration-300">
        <Link to={`/class/${classId}`}>
          <img
            src={classImage}
            alt="class"
            className="h-48 w-full object-cover border-b-2 border-gray-300"
          />
        </Link>
        <div className="p-4">
          <p className="text-gray-600 text-sm">
            Semester {classInfo?.classItem?.semester % 10} | Academic Year{" "}
            {classInfo?.course?.startAcademicYear}-
            {classInfo?.course?.endAcademicYear}
          </p>
          <Link
            to={`/class/${classId}`}
            className="block mt-1 text-lg font-semibold text-[#26597C] hover:underline line-clamp-2"
          >
            {classInfo?.subject?.code}_{classInfo?.subject?.name}
          </Link>
          <p className="text-sm italic text-gray-500">
            [{classInfo.classItem?.name}]
          </p>
        </div>
      </div>
    );
  } else {
    return (
      <div className="w-full flex gap-5 border-2 border-teal-600 hover:border-[3px] overflow-hidden rounded-lg">
        <Link to={`/class/${classId}`}>
          <img
            src={classImage}
            alt="class image"
            className="h-[200px] w-[350px] border-r-2 border-teal-600"
          />
        </Link>
        <div className="p-3 flex flex-col gap-1 justify-center">
          <p className="text-lg">
            Semester {classInfo?.classItem?.semester % 10} | Academic Year{" "}
            {classInfo?.course?.startAcademicYear} -{" "}
            {classInfo?.course?.endAcademicYear}
          </p>
          <Link to={`/class/${classId}`} className="hover:underline">
            <span className="text-2xl font-semibold line-clamp-2">
              {classInfo?.subject?.code}_{classInfo?.subject?.name}
            </span>
          </Link>
          <span className="italic text-lg">[{classInfo.classItem?.name}]</span>
        </div>
      </div>
    );
  }
}

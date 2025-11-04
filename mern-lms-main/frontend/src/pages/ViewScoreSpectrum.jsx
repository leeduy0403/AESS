import { useEffect, useState } from "react";
import { Link, useLocation, useParams } from "react-router-dom";
import ClassSidebar from "../components/ClassSidebar";
import { Accordion, AccordionSummary, Typography } from "@mui/material";
import moment from "moment";
import { useDispatch, useSelector } from "react-redux";
import { toggleIsEditMode } from "../redux/isEditMode/isEditModeSlice";
import { BarChart } from "@mui/x-charts/BarChart";
import {
  ExpandMore as ExpandMoreIcon,
  Assignment as AssignmentIcon,
  Person as PersonIcon,
  Groups as GroupsIcon,
  ArrowBack as ArrowBackIcon,
} from "@mui/icons-material";

export default function ViewScoreSpectrum() {
  const dispatch = useDispatch();
  const { currentUser } = useSelector((state) => state.user);
  const { classId } = useParams();
  const { isEditMode } = useSelector((state) => state.isEditMode);
  const { tabIndex } = useSelector((state) => state.tabIndex);
  const location = useLocation();
  const [tabValue, setTabValue] = useState("");
  const [assignmentId, setAssignmentId] = useState("");
  const [assignment, setAssignment] = useState([]);
  const [classInfo, setClassInfo] = useState({});
  const [viewSubmissionsResponse, setViewSubmissionsResponse] = useState([]);
  const [bins, setBins] = useState([]);
  const [stats, setStats] = useState({
    total: 0,
    average: 0,
    median: 0,
    highest: 0,
    perfect10s: 0,
    stdDev: 0,
    distribution: [],
  });
  const [chartData, setChartData] = useState([]);
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
    if (tabIndex === 0) {
      setTabValue("Material");
    }
    if (tabIndex === 1) {
      setTabValue("Group");
    }
    if (tabIndex === 2) {
      setTabValue("Assignment");
    }
    if (tabIndex === 3) {
      setTabValue("Grade");
    }
    if (tabIndex === 4) {
      setTabValue("Forum");
    }
  }, [tabIndex]);

  useEffect(() => {
    const urlParams = new URLSearchParams(location.search);
    const assignmentIdFromUrl = urlParams.get("assignmentId");
    if (assignmentIdFromUrl) {
      setAssignmentId(assignmentIdFromUrl);
    }
  }, [location.search]);

  useEffect(() => {
    const fetchMaterials = async () => {
      if (!assignmentId) {
        return;
      }
      try {
        const resAssignment = await fetch(
          `/api/assignment/get/${classId}?assignmentId=${assignmentId}`
        );
        const assignment = await resAssignment.json();
        if (resAssignment.ok) {
          setAssignment(assignment?.assignments[0]);
        }
      } catch (error) {
        console.log(error.message);
      }
    };
    fetchMaterials();
  }, [assignmentId, classId]);

  useEffect(() => {
    const stringToDecimal = (str) => {
      if (!str || typeof str !== "string") {
        return 1;
      }
      let sum = 0;
      for (let i = 0; i < str.length; i++) {
        sum += str.charCodeAt(i);
      }
      const normalized = sum % 1000 || 1;
      return Math.max(1, Math.floor(normalized));
    };
    const fetchClassInfo = async () => {
      try {
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
      } catch (error) {
        console.log(error.message);
      }
    };
    fetchClassInfo();
  }, [classId]);

  useEffect(() => {
    const fetchSubmissionsInfo = async () => {
      if (!assignmentId) {
        return;
      }
      try {
        const res = await fetch(
          `/api/assignment/view-submissions/${classId}/${assignmentId}`
        );
        const data = await res.json();
        if (res.ok) {
          setViewSubmissionsResponse(data);
        }
      } catch (error) {
        console.log(error.message);
      }
    };
    fetchSubmissionsInfo();
  }, [classId, assignmentId]);

  useEffect(() => {
    if (!viewSubmissionsResponse || viewSubmissionsResponse.length === 0)
      return;
    const allScores = [];
    const coefficients = assignment?.coefficients || [];
    viewSubmissionsResponse.forEach((item) => {
      const individualScoresObj = item?.lastSubmissionInfo?.individualScores;
      const coeffs = item?.lastSubmissionInfo?.coefficients || coefficients;
      const calculateScore = (scoresArray) => {
        if (Array.isArray(scoresArray) && scoresArray?.length > 0) {
          return scoresArray
            .slice(0, coeffs?.length)
            .reduce(
              (sum, score, idx) => sum + (score || 0) * (coeffs[idx] || 1),
              0
            );
        }
        return null;
      };
      if (assignment?.type === "Individual") {
        const weightedScore = calculateScore(individualScoresObj);
        if (typeof weightedScore === "number") allScores.push(weightedScore);
      }
      if (assignment?.type === "Group") {
        if (individualScoresObj && typeof individualScoresObj === "object") {
          Object.values(individualScoresObj).forEach((scoreArray) => {
            const weightedScore = calculateScore(scoreArray);
            if (typeof weightedScore === "number")
              allScores.push(weightedScore);
          });
        }
      }
    });

    if (allScores.length > 0) {
      const maxScore = Math.max(...allScores);
      const binCount = Math.ceil(maxScore);
      const binsArray = Array.from(
        { length: binCount },
        (_, i) => `${i}-${i + 1}`
      );
      const countsArray = new Array(binCount).fill(0);
      allScores.forEach((score) => {
        const binIndex = Math.min(Math.floor(score), binCount - 1);
        countsArray[binIndex]++;
      });
      setChartData(countsArray);
      setBins(binsArray);
      const sortedScores = [...allScores].sort((a, b) => a - b);
      const total = sortedScores.length;
      const average = sortedScores.reduce((sum, val) => sum + val, 0) / total;
      const median =
        total % 2 === 0
          ? (sortedScores[total / 2 - 1] + sortedScores[total / 2]) / 2
          : sortedScores[Math.floor(total / 2)];
      const stdDev = Math.sqrt(
        sortedScores.reduce((sum, val) => sum + Math.pow(val - average, 2), 0) /
          total
      );
      const highest = sortedScores[sortedScores.length - 1];
      setStats({
        total,
        average,
        median,
        stdDev,
        highest,
      });
    }
  }, [viewSubmissionsResponse, assignment?.type, assignment?.coefficients]);

  return (
    <div className="min-h-screen flex flex-col mx-auto lg:w-10/12 mb-40">
      <div className="h-[240px] my-5 flex border-2 border-gray-300 rounded-xl overflow-hidden shadow-md">
        <img
          src={classImage}
          alt="class cover"
          className="w-[23vw] h-full border-r-2 border-gray-300"
        />
        <div className="p-12 flex flex-col gap-1 justify-center">
          <p className="text-xl font-bold">
            Semester {classInfo?.classItem?.semester % 10} | Academic Year{" "}
            {classInfo?.course?.startAcademicYear} -{" "}
            {classInfo?.course?.endAcademicYear}
          </p>
          <span className="text-3xl font-bold ">
            {classInfo?.subject?.code}_{classInfo?.subject?.name}
          </span>
          <div className="flex gap-2">
            <span className="text-xl text-cyan-600">Class:</span>
            <span className="text-xl text-gray-950">
              {classInfo?.classItem?.name}
            </span>
          </div>
          <div className="flex gap-2">
            <span className="text-xl text-cyan-600">Educators:</span>
            {classInfo?.educators?.length > 0 &&
              classInfo?.educators.map((educator, i) => (
                <span className="text-xl text-gray-950" key={i}>
                  {i !== classInfo?.educators?.length - 1
                    ? `${educator?.name},`
                    : educator?.name}
                </span>
              ))}
          </div>
          {currentUser?.isEducator && (
            <div className="flex gap-3 items-center">
              <span className="text-xl font-semibold">Edit Mode</span>
              {isEditMode ? (
                <i
                  className="fa-solid fa-toggle-on fa-xl cursor-pointer"
                  onClick={() => dispatch(toggleIsEditMode())}
                ></i>
              ) : (
                <i
                  className="fa-solid fa-toggle-off fa-xl cursor-pointer"
                  onClick={() => dispatch(toggleIsEditMode())}
                ></i>
              )}
            </div>
          )}
        </div>
      </div>
      <div className="flex flex-row">
        <div className="basis-9/12">
          <Link to={`/class/${classId}`}>
            <div className="flex gap-2 items-center pb-4 hover:underline text-cyan-600 font-semibold">
              <ArrowBackIcon />
              <span>Back to {tabValue} tab</span>
            </div>
          </Link>
          <div className="flex flex-col gap-4">
            <Accordion
              defaultExpanded
              className="mb-4"
              sx={{
                borderRadius: 2,
                overflow: "hidden",
              }}
            >
              <AccordionSummary
                expandIcon={<ExpandMoreIcon sx={{ color: "white" }} />}
                aria-controls="panel1-content"
                id="panel1-header"
                sx={{
                  backgroundColor: "#26597C",
                  color: "#ffffff",
                  borderTopLeftRadius: 12,
                  borderTopRightRadius: 12,
                }}
              >
                <div className="flex gap-2 items-center">
                  <AssignmentIcon />
                  <Typography
                    component="span"
                    style={{ fontSize: "18px", fontWeight: "bold" }}
                  >
                    {assignment?.title}
                  </Typography>
                  {assignment?.type === "Individual" ? (
                    <PersonIcon />
                  ) : (
                    <GroupsIcon />
                  )}
                </div>
              </AccordionSummary>
              <div style={{ backgroundColor: "#F8F8D5", padding: "4px" }}>
                <div className="flex gap-2 my-5 lg:w-11/12 mx-auto">
                  <div className="flex flex-col gap-2">
                    <div className="font-bold">Open Date: </div>
                    <div className="font-bold">Due Date: </div>
                    <div className="font-bold">Description: </div>
                  </div>
                  <div className="flex flex-col gap-2">
                    <div className="font-bold text-red-600">
                      {assignment?.startDate
                        ? moment(assignment?.startDate).format(
                            "HH:mm:ss DD/MM/YYYY"
                          )
                        : "---"}
                    </div>
                    <div className="font-bold text-red-600">
                      {assignment?.endDate
                        ? moment(assignment?.endDate).format(
                            "HH:mm:ss DD/MM/YYYY"
                          )
                        : "---"}
                    </div>
                    <div className="">{assignment?.description || "---"}</div>
                  </div>
                </div>
              </div>
            </Accordion>
            <div className="font-bold text-3xl text-center">Score Spectrum</div>
            <BarChart
              series={[{ data: chartData }]}
              height={290}
              xAxis={[{ data: bins, scaleType: "band" }]}
              yAxis={[
                {
                  min: 0,
                  max: Math.max(...chartData) + 1,
                  tickMinStep: 1,
                  label: "Number of Students",
                },
              ]}
              margin={{ top: 10, bottom: 30, left: 40, right: 10 }}
            />
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="bg-[#26597C] text-white">
                    <th className="p-4 border border-black">
                      Total number of submissions
                    </th>
                    <th className="p-4 border border-black">Average Score</th>
                    <th className="p-4 border border-black">Median</th>
                    <th className="p-4 border border-black">
                      Standard Deviation
                    </th>
                    <th className="p-4 border border-black">
                      Highest score student achieved
                    </th>
                  </tr>
                </thead>
                <tbody>
                  <tr className="text-gray-950">
                    <td className="p-4 border border-black text-center">
                      {stats?.total}
                    </td>
                    <td className="p-4 border border-black text-center">
                      {stats?.average.toFixed(2)}
                    </td>
                    <td className="p-4 border border-black text-center">
                      {stats?.median.toFixed(2)}
                    </td>
                    <td className="p-4 border border-black text-center">
                      {stats?.stdDev.toFixed(2)}
                    </td>
                    <td className="p-4 border border-black text-center">
                      {stats?.highest}
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        </div>
        <div className="basis-3/12 ml-5">
          <ClassSidebar classId={classId} />
        </div>
      </div>
    </div>
  );
}

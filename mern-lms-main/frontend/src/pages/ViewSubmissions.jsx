import React, { useEffect, useState } from "react";
import { Link, useLocation, useParams } from "react-router-dom";
import ClassSidebar from "../components/ClassSidebar";
import {
  Accordion,
  AccordionSummary,
  Alert,
  Button,
  Typography,
} from "@mui/material";
import moment from "moment";
import { Avatar, Modal, TextInput, Spinner } from "flowbite-react";
import { HiOutlineExclamationCircle } from "react-icons/hi";
import dayjs from "dayjs";
import { useDispatch, useSelector } from "react-redux";
import { toggleIsEditMode } from "../redux/isEditMode/isEditModeSlice";
import {
  Assignment as AssignmentIcon,
  ExpandMore as ExpandMoreIcon,
  Groups as GroupsIcon,
  Person as PersonIcon,
  ArrowBack as ArrowBackIcon,
} from "@mui/icons-material";

export default function ViewSubmissions() {
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
  const [
    showModalUpdateAssignmentIsScorePublishTrue,
    setShowModalUpdateAssignmentIsScorePublishTrue,
  ] = useState(false);
  const [
    showModalUpdateAssignmentIsScorePublishFalse,
    setShowModalUpdateAssignmentIsScorePublishFalse,
  ] = useState(false);
  const [showModalApproveAISuggestion, setShowModalApproveAISuggestion] =
    useState(false);
  const [showModalGenerateAISuggestion, setShowModalGenerateAISuggestion] =
    useState(false);
  const [formData, setFormData] = useState({});
  const [updateError, setUpdateError] = useState(null);
  const [updateSuccess, setUpdateSuccess] = useState(null);
  const [loading, setLoading] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);
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
    const initialFormData = {};
    if (assignment?.type === "Individual") {
      viewSubmissionsResponse?.length > 0 &&
        viewSubmissionsResponse.map((item, i) => {
          const submissionId = item?.lastSubmissionInfo?._id;
          let initialScore;
          if (item?.lastSubmissionInfo?.individualScores?.length > 0) {
            initialScore = item?.lastSubmissionInfo?.individualScores || 0;
          }
          if (!initialFormData?.[submissionId]) {
            initialFormData[submissionId] = [];
          }
          initialFormData[submissionId] = initialScore;
        });
    } else if (assignment?.type === "Group") {
      viewSubmissionsResponse?.length > 0 &&
        viewSubmissionsResponse.map(
          (item) =>
            item?.groupInfo?.members?.length > 0 &&
            item?.groupInfo?.members.map((member) => {
              const studentId = member?._id;
              const submissionId = item?.lastSubmissionInfo?._id;
              const initialScore =
                item?.lastSubmissionInfo?.individualScores?.[member?._id] || 0;

              if (!initialFormData?.[submissionId]) {
                initialFormData[submissionId] = {};
              }
              initialFormData[submissionId][studentId] = initialScore;
            })
        );
    }
    setFormData(initialFormData);
  }, [assignment?.type, viewSubmissionsResponse]);

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
          setAssignment(assignment.assignments[0]);
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

  useEffect(() => {
    fetchSubmissionsInfo();
  }, [classId, assignmentId]);

  const handleChangeIndividual = (e, j) => {
    setFormData((prevFormData) => {
      const updatedScores = [...(prevFormData[e.target.id] || [])];
      updatedScores[j] = Number(e.target.value);
      return {
        ...prevFormData,
        [e.target.id]: updatedScores,
      };
    });
  };

  const handleChangeGroup = (e, j, memberId) => {
    setFormData((prevFormData) => {
      const submissionId = e.target.id;
      const prevSubmission = prevFormData[submissionId] || {};
      let prevStudentScores = prevSubmission[memberId];
      prevStudentScores[j] = Number(e.target.value);
      return {
        ...prevFormData,
        [submissionId]: {
          ...prevSubmission,
          [memberId]: prevStudentScores,
        },
      };
    });
  };

  const handleUpdateAssignmentIsScorePublishTrue = async () => {
    setShowModalUpdateAssignmentIsScorePublishTrue(false);
    setLoading(true);
    setUpdateError(null);
    setUpdateSuccess(null);
    try {
      const res = await fetch(
        `/api/assignment/update/${classId}/${assignmentId}`,
        {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            isScorePublish: true,
            publishDate: dayjs(Date.now()),
          }),
        }
      );
      const data = await res.json();
      if (!res.ok) {
        setUpdateError(data.message);
      } else {
        setUpdateSuccess("Publish score successfully!");
      }
      setLoading(false);
    } catch (error) {
      setUpdateError(error.message);
      setLoading(false);
    }
  };

  const handleUpdateAssignmentIsScorePublishFalse = async () => {
    setShowModalUpdateAssignmentIsScorePublishFalse(false);
    setLoading(true);
    setUpdateError(null);
    setUpdateSuccess(null);
    try {
      const res = await fetch(
        `/api/assignment/update/${classId}/${assignmentId}`,
        {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            isScorePublish: false,
          }),
        }
      );
      const data = await res.json();
      if (!res.ok) {
        setUpdateError(data.message);
      } else {
        setUpdateSuccess("Unpublish score successfully!");
      }
      setLoading(false);
    } catch (error) {
      setUpdateError(error.message);
      setLoading(false);
    }
  };

  const handleApproveAISuggestion = async () => {
    setShowModalApproveAISuggestion(false);
    setLoading(true);
    setUpdateError(null);
    setUpdateSuccess(null);
    try {
      const res = await fetch(`/api/assignment/approve/${assignmentId}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
      });
      const data = await res.json();
      if (!res.ok) {
        setUpdateError(data.message);
      } else {
        setUpdateSuccess("Approve AI score successfully!");
        fetchSubmissionsInfo();
      }
      setLoading(false);
    } catch (error) {
      setUpdateError(error.message);
      setLoading(false);
    }
  };

  const handleGenerateAISuggestion = async () => {
    setShowModalGenerateAISuggestion(false);
    setUpdateError(null);
    setUpdateSuccess(null);
    setIsGenerating(true);
    try {
      const res = await fetch(
        `/api/assignment/save-results/${assignment?.classId}/${assignment?._id}`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
        }
      );
      const data = await res.json();
      if (!res.ok) {
        setUpdateError(data.message);
      } else {
        setUpdateSuccess("Generate AI score successfully!");
        fetchSubmissionsInfo();
      }
      setLoading(false);
      setIsGenerating(false);
    } catch (error) {
      setUpdateError(error.message);
      setIsGenerating(false);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setUpdateError(null);
    setUpdateSuccess(null);
    try {
      const res = await fetch(`/api/submission/update-overallScores`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          formData,
        }),
      });
      const data = await res.json();
      if (!res.ok) {
        setUpdateError(data.message);
      } else {
        setUpdateSuccess("Update overall score for student(s) successfully!");
        fetchSubmissionsInfo();
      }
      setLoading(false);
    } catch (error) {
      setUpdateError(error.message);
      setLoading(false);
    }
  };

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
          <div className="flex flex-col">
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
                    <div className="font-bold">Max Attempt: </div>
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
                    <div className="font-semibold">
                      {assignment?.maxAttempt}
                    </div>
                    <div className="">{assignment?.description || "---"}</div>
                  </div>
                </div>
              </div>
            </Accordion>
            <div className="flex justify-between sticky top-0 bg-white/80 py-4 z-10">
              <div className="flex gap-4">
                <Link
                  to={`/class/${classId}/view-score-spectrum?assignmentId=${assignmentId}`}
                >
                  <Button
                    variant="contained"
                    component="label"
                    style={{
                      backgroundColor: "#26597C",
                      color: "#ffffff",
                      textTransform: "none",
                    }}
                    size="large"
                    disabled={loading}
                  >
                    View score spectrum
                  </Button>
                </Link>
                {assignment?.isScorePublish ? (
                  <Button
                    variant="contained"
                    component="label"
                    style={{
                      backgroundColor: "#26597C",
                      color: "#ffffff",
                      textTransform: "none",
                    }}
                    size="large"
                    onClick={(e) => {
                      setShowModalUpdateAssignmentIsScorePublishFalse(true);
                    }}
                    disabled={loading}
                  >
                    Unpublish score
                  </Button>
                ) : (
                  <Button
                    variant="contained"
                    component="label"
                    style={{
                      backgroundColor: "#26597C",
                      color: "#ffffff",
                      textTransform: "none",
                    }}
                    size="large"
                    onClick={(e) => {
                      setShowModalUpdateAssignmentIsScorePublishTrue(true);
                    }}
                    disabled={loading}
                  >
                    Publish score
                  </Button>
                )}
              </div>
              <div className="flex gap-4">
                <Button
                  variant="contained"
                  component="label"
                  style={{
                    backgroundColor: "#26597C",
                    color: "#ffffff",
                    textTransform: "none",
                  }}
                  size="large"
                  onClick={(e) => {
                    setShowModalGenerateAISuggestion(true);
                  }}
                  disabled={loading || isGenerating}
                >
                  {isGenerating ? (
                    <>
                      <Spinner size="sm" />
                      <span className="pl-3">Generating...</span>
                    </>
                  ) : (
                    "Generate AI score"
                  )}
                </Button>
                <Button
                  variant="contained"
                  component="label"
                  style={{
                    backgroundColor: "#26597C",
                    color: "#ffffff",
                    textTransform: "none",
                  }}
                  size="large"
                  onClick={(e) => {
                    setShowModalApproveAISuggestion(true);
                  }}
                  disabled={loading}
                >
                  Approve AI score
                </Button>
              </div>
            </div>
            {updateError && (
              <Alert severity="error" className="border border-red-600 mb-4">
                {updateError}
              </Alert>
            )}
            {updateSuccess && (
              <Alert
                severity="success"
                className="border border-green-600 mb-4"
              >
                {updateSuccess}
              </Alert>
            )}
            {assignment?.type === "Individual" ? (
              <div className="overflow-x-auto shadow-md rounded-lg">
                <table className="w-full text-sm bg-white border border-gray-300">
                  <thead className="text-white bg-[#26597C]">
                    <tr className="text-center">
                      <th className="p-4 border">Student Name</th>
                      <th className="p-4 border">Score Component</th>
                      <th className="p-4 border">Score</th>
                      <th className="p-4 border">AI Score</th>
                    </tr>
                  </thead>
                  <tbody>
                    {viewSubmissionsResponse?.length > 0 &&
                      viewSubmissionsResponse.map((item, i) => (
                        <tr
                          key={i}
                          className={`transition-all ${
                            i % 2 === 0 ? "bg-white" : "bg-[#F8F8D5]"
                          }`}
                        >
                          <td className="p-4 border border-gray-300">
                            <div className="flex gap-2 items-center justify-start">
                              <Avatar
                                alt="User avatar"
                                img={item?.studentInfo?.profilePicture}
                                size="sm"
                                rounded
                              />
                              <div className="">
                                {item?.studentInfo?.studentId}
                              </div>
                              <Link
                                to={`/class/${classId}/view-attempts?assignmentId=${assignmentId}&studentId=${item?.studentInfo?._id}`}
                                className="hover:underline text-cyan-600"
                              >
                                {item?.studentInfo?.name}
                              </Link>
                            </div>
                          </td>
                          <td className="p-4 border border-gray-300">
                            <div className="grid grid-cols-[auto_auto] gap-4">
                              {item?.lastSubmissionInfo?.scoreComponent
                                ?.length > 0 &&
                                item?.lastSubmissionInfo?.scoreComponent.map(
                                  (e, j) => (
                                    <React.Fragment key={j}>
                                      <div>
                                        (Coef: x
                                        {item?.lastSubmissionInfo
                                          ?.coefficients?.[j] || 1}
                                        ) {e}
                                      </div>
                                      <div>
                                        {isEditMode ? (
                                          <TextInput
                                            required
                                            sizing="sm"
                                            id={item?.lastSubmissionInfo?._id}
                                            value={
                                              formData?.[
                                                item?.lastSubmissionInfo?._id
                                              ]?.[j] ??
                                              item?.lastSubmissionInfo
                                                ?.individualScores?.[j] ??
                                              ""
                                            }
                                            onChange={(e) =>
                                              handleChangeIndividual(e, j)
                                            }
                                          />
                                        ) : (
                                          <div className="font-semibold">
                                            {
                                              item?.lastSubmissionInfo
                                                ?.individualScores?.[j]
                                            }
                                          </div>
                                        )}
                                      </div>
                                    </React.Fragment>
                                  )
                                )}
                            </div>
                          </td>
                          <td className="p-4 border border-gray-300 text-center">
                            <div className="font-semibold">
                              {formData?.[item?.lastSubmissionInfo?._id]
                                ?.length > 1
                                ? formData?.[item?.lastSubmissionInfo?._id]
                                    ?.slice(0, -1)
                                    ?.reduce(
                                      (sum, score, idx) =>
                                        sum +
                                        score *
                                          (item?.lastSubmissionInfo
                                            ?.coefficients?.[idx] || 1),
                                      0
                                    )
                                : item?.lastSubmissionInfo?.overallScore}
                            </div>
                          </td>
                          <td className="p-4 border border-gray-300 text-center">
                            <div className="font-semibold">
                              {item?.lastSubmissionInfo?.overallAIScore}
                            </div>
                          </td>
                        </tr>
                      ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <div className="overflow-x-auto shadow-md rounded-lg">
                <table className="w-full text-sm bg-white border border-gray-300">
                  <thead className="text-white bg-[#26597C]">
                    <tr className="text-center">
                      <th className="p-4 border">Group</th>
                      <th className="p-4 border">Student Name</th>
                      <th className="p-4 border">Score Component</th>
                      <th className="p-4 border">Score</th>
                      <th className="p-4 border">Score from AI</th>
                    </tr>
                  </thead>
                  <tbody>
                    {viewSubmissionsResponse?.length > 0 &&
                      viewSubmissionsResponse.map(
                        (item, i) =>
                          item?.groupInfo?.members?.length > 0 &&
                          item?.groupInfo?.members.map((member, index) => (
                            <tr
                              key={index}
                              className={`transition-all ${
                                i % 2 === 0 ? "bg-white" : "bg-[#F8F8D5]"
                              }`}
                            >
                              {index === 0 && (
                                <td
                                  className="p-4 border border-gray-300 font-medium text-sky-700 text-center"
                                  rowSpan={item?.groupInfo?.members?.length}
                                >
                                  {item?.groupInfo?.name}
                                </td>
                              )}
                              <td className="p-4 border border-gray-300">
                                <div className="flex gap-2 items-center justify-start">
                                  <Avatar
                                    alt="User avatar"
                                    img={member?.profilePicture}
                                    size="sm"
                                    rounded
                                  />
                                  <div className="">{member?.studentId}</div>
                                  <Link
                                    to={`/class/${classId}/view-attempts?assignmentId=${assignmentId}&studentId=${member?._id}`}
                                    className="hover:underline text-cyan-600"
                                  >
                                    {member?.name}
                                  </Link>
                                </div>
                              </td>
                              <td className="p-4 border border-gray-300">
                                <div className="grid grid-cols-[auto_auto] gap-4">
                                  {item?.lastSubmissionInfo?.scoreComponent
                                    ?.length > 0 &&
                                    item?.lastSubmissionInfo?.scoreComponent.map(
                                      (e, j) => (
                                        <React.Fragment key={j}>
                                          <div>
                                            (Coef: x
                                            {
                                              item?.lastSubmissionInfo
                                                ?.coefficients?.[j]
                                            }
                                            ) {e}
                                          </div>
                                          <div>
                                            {isEditMode ? (
                                              <TextInput
                                                required
                                                sizing="sm"
                                                id={
                                                  item?.lastSubmissionInfo?._id
                                                }
                                                value={
                                                  formData?.[
                                                    item?.lastSubmissionInfo
                                                      ?._id
                                                  ]?.[member?._id]?.[j] ??
                                                  item?.lastSubmissionInfo
                                                    ?.individualScores?.[
                                                    member?._id
                                                  ]?.[j] ??
                                                  ""
                                                }
                                                onChange={(e) =>
                                                  handleChangeGroup(
                                                    e,
                                                    j,
                                                    member?._id
                                                  )
                                                }
                                              />
                                            ) : (
                                              <div className="font-semibold">
                                                {
                                                  item?.lastSubmissionInfo
                                                    ?.individualScores?.[
                                                    member?._id
                                                  ]?.[j]
                                                }
                                              </div>
                                            )}
                                          </div>
                                        </React.Fragment>
                                      )
                                    )}
                                </div>
                              </td>
                              <td className="p-4 border border-gray-300 text-center">
                                <div className="font-semibold">
                                  {formData?.[item?.lastSubmissionInfo?._id]?.[
                                    member?._id
                                  ]?.length > 1
                                    ? formData?.[
                                        item?.lastSubmissionInfo?._id
                                      ]?.[member?._id]
                                        ?.slice(0, -1)
                                        ?.reduce(
                                          (sum, score, idx) =>
                                            sum +
                                            score *
                                              (item?.lastSubmissionInfo
                                                ?.coefficients?.[idx] || 1),
                                          0
                                        )
                                    : item?.lastSubmissionInfo?.overallScore}
                                </div>
                              </td>
                              <td className="p-4 border border-gray-300 text-center">
                                <div className="font-semibold">
                                  {item?.lastSubmissionInfo?.overallAIScore}
                                </div>
                              </td>
                            </tr>
                          ))
                      )}
                  </tbody>
                </table>
              </div>
            )}
            <div className="flex gap-4 justify-end my-4">
              <Button
                variant="contained"
                component="label"
                style={{
                  backgroundColor: "#26597C",
                  color: "#ffffff",
                  textTransform: "none",
                }}
                type="submit"
                size="large"
                onClick={handleSubmit}
                disabled={loading}
              >
                Save changes
              </Button>
            </div>
            {updateError && (
              <Alert severity="error" className="mb-4 border border-red-600">
                {updateError}
              </Alert>
            )}
            {updateSuccess && (
              <Alert
                severity="success"
                className="mb-4 border border-green-600"
              >
                {updateSuccess}
              </Alert>
            )}
          </div>
        </div>
        <div className="basis-3/12 ml-5">
          <ClassSidebar classId={classId} />
        </div>
      </div>
      <Modal
        show={showModalUpdateAssignmentIsScorePublishTrue}
        onClose={() => setShowModalUpdateAssignmentIsScorePublishTrue(false)}
        popup
        size="md"
      >
        <Modal.Header />
        <Modal.Body>
          <div className="text-center">
            <HiOutlineExclamationCircle className="h-14 w-14 text-red-600 dark:text-gray-200 mb-4 mx-auto" />
            <h3 className="mb-5 text-lg">
              Are you sure you want to publish scores of this assignment?
            </h3>
            <div className="flex justify-center gap-4">
              <Button
                variant="contained"
                component="label"
                style={{
                  backgroundColor: "oklch(0.577 0.245 27.325)",
                  color: "#ffffff",
                  textTransform: "none",
                }}
                onClick={handleUpdateAssignmentIsScorePublishTrue}
                fullWidth
              >
                Yes, I'm sure
              </Button>
              <Button
                variant="contained"
                component="label"
                style={{
                  color: "#ffffff",
                  textTransform: "none",
                }}
                onClick={() =>
                  setShowModalUpdateAssignmentIsScorePublishTrue(false)
                }
                fullWidth
              >
                No, cancel
              </Button>
            </div>
          </div>
        </Modal.Body>
      </Modal>
      <Modal
        show={showModalUpdateAssignmentIsScorePublishFalse}
        onClose={() => setShowModalUpdateAssignmentIsScorePublishFalse(false)}
        popup
        size="md"
      >
        <Modal.Header />
        <Modal.Body>
          <div className="text-center">
            <HiOutlineExclamationCircle className="h-14 w-14 text-red-600 dark:text-gray-200 mb-4 mx-auto" />
            <h3 className="mb-5 text-lg">
              Are you sure you want to unpublish scores of this assignment?
            </h3>
            <div className="flex justify-center gap-4">
              <Button
                variant="contained"
                component="label"
                style={{
                  backgroundColor: "oklch(0.577 0.245 27.325)",
                  color: "#ffffff",
                  textTransform: "none",
                }}
                onClick={handleUpdateAssignmentIsScorePublishFalse}
                fullWidth
              >
                Yes, I'm sure
              </Button>
              <Button
                variant="contained"
                component="label"
                style={{
                  color: "#ffffff",
                  textTransform: "none",
                }}
                onClick={() =>
                  setShowModalUpdateAssignmentIsScorePublishFalse(false)
                }
                fullWidth
              >
                No, cancel
              </Button>
            </div>
          </div>
        </Modal.Body>
      </Modal>
      <Modal
        show={showModalApproveAISuggestion}
        onClose={() => setShowModalApproveAISuggestion(false)}
        popup
        size="md"
      >
        <Modal.Header />
        <Modal.Body>
          <div className="text-center">
            <HiOutlineExclamationCircle className="h-14 w-14 text-red-600 dark:text-gray-200 mb-4 mx-auto" />
            <h3 className="mb-5 text-lg">
              Are you sure you want to approve AI score?
            </h3>
            <div className="flex justify-center gap-4">
              <Button
                variant="contained"
                component="label"
                style={{
                  backgroundColor: "oklch(0.577 0.245 27.325)",
                  color: "#ffffff",
                  textTransform: "none",
                }}
                onClick={handleApproveAISuggestion}
                fullWidth
              >
                Yes, I'm sure
              </Button>
              <Button
                variant="contained"
                component="label"
                style={{
                  color: "#ffffff",
                  textTransform: "none",
                }}
                onClick={() => setShowModalApproveAISuggestion(false)}
                fullWidth
              >
                No, cancel
              </Button>
            </div>
          </div>
        </Modal.Body>
      </Modal>
      <Modal
        show={showModalGenerateAISuggestion}
        onClose={() => setShowModalGenerateAISuggestion(false)}
        popup
        size="md"
      >
        <Modal.Header />
        <Modal.Body>
          <div className="text-center">
            <HiOutlineExclamationCircle className="h-14 w-14 text-red-600 dark:text-gray-200 mb-4 mx-auto" />
            <h3 className="mb-5 text-lg">
              Are you sure you want to generate AI score?
            </h3>
            <div className="flex justify-center gap-4">
              <Button
                variant="contained"
                component="label"
                style={{
                  backgroundColor: "oklch(0.577 0.245 27.325)",
                  color: "#ffffff",
                  textTransform: "none",
                }}
                onClick={handleGenerateAISuggestion}
                fullWidth
              >
                Yes, I'm sure
              </Button>
              <Button
                variant="contained"
                component="label"
                style={{
                  color: "#ffffff",
                  textTransform: "none",
                }}
                onClick={() => setShowModalApproveAISuggestion(false)}
                fullWidth
              >
                No, cancel
              </Button>
            </div>
          </div>
        </Modal.Body>
      </Modal>
    </div>
  );
}

import { useEffect, useState } from "react";
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
import { useDispatch, useSelector } from "react-redux";
import pdf from "../assets/pdf.png";
import { toggleIsEditMode } from "../redux/isEditMode/isEditModeSlice";
import { Avatar, Modal, Spinner, Textarea, TextInput } from "flowbite-react";
import { HiOutlineExclamationCircle } from "react-icons/hi";
import {
  Assignment as AssignmentIcon,
  ExpandMore as ExpandMoreIcon,
  Folder as FolderIcon,
  Groups as GroupsIcon,
  Person as PersonIcon,
  School as SchoolIcon,
  ArrowBack as ArrowBackIcon,
} from "@mui/icons-material";

export default function ViewAttempts() {
  const dispatch = useDispatch();
  const { currentUser } = useSelector((state) => state.user);
  const { classId } = useParams();
  const { isEditMode } = useSelector((state) => state.isEditMode);
  const { tabIndex } = useSelector((state) => state.tabIndex);
  const location = useLocation();
  const [tabValue, setTabValue] = useState("");
  const [assignmentId, setAssignmentId] = useState("");
  const [studentId, setStudentId] = useState("");
  const [assignment, setAssignment] = useState([]);
  const [classInfo, setClassInfo] = useState({});
  const [submissionsInfo, setSubmissionsInfo] = useState([]);
  const [lastSubmissionInfo, setLastSubmissionInfo] = useState({});
  const [userGroupInfo, setUserGroupInfo] = useState({});
  const [topicId, setTopicId] = useState(null);
  const [questionsInfo, setQuestionsInfo] = useState([]);
  const [questionFormData, setQuestionFormData] = useState("");
  const [replyFormData, setReplyFormData] = useState("");
  const [questionIdToReply, setQuestionIdToReply] = useState(null);
  const [isShowAddQuestion, setIsShowAddQuestion] = useState(false);
  const [addQuestionError, setAddQuestionError] = useState(null);
  const [addQuestionSuccess, setAddQuestionSuccess] = useState(null);
  const [addReplyError, setAddReplyError] = useState(null);
  const [addReplySuccess, setAddReplySuccess] = useState(null);
  const [formData, setFormData] = useState({});
  const [updateError, setUpdateError] = useState(null);
  const [updateSuccess, setUpdateSuccess] = useState(null);
  const [approveError, setApproveError] = useState(null);
  const [approveSuccess, setApproveSuccess] = useState(null);
  const [generateError, setGenerateError] = useState(null);
  const [generateSuccess, setGenerateSuccess] = useState(null);
  const [loading, setLoading] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);
  const [showModalApproveAISuggestion, setShowModalApproveAISuggestion] =
    useState(false);
  const [showModalGenerateAIScore, setShowModalGenerateAIScore] =
    useState(false);
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
    const studentIdFromUrl = urlParams.get("studentId");
    if (assignmentIdFromUrl) {
      setAssignmentId(assignmentIdFromUrl);
    }
    if (studentIdFromUrl) {
      setStudentId(studentIdFromUrl);
    }
  }, [location.search]);

  useEffect(() => {
    const initialFormData = {};
    if (assignment?.type === "Individual") {
      const submissionId = lastSubmissionInfo?._id;
      let initialScore;
      if (lastSubmissionInfo?.individualScores?.length > 0) {
        initialScore = lastSubmissionInfo?.individualScores || 0;
      }
      if (!initialFormData[submissionId]) {
        initialFormData[submissionId] = [];
      }
      initialFormData[submissionId] = initialScore;
    } else if (assignment?.type === "Group") {
      userGroupInfo?.members?.length > 0 &&
        userGroupInfo?.members.map((member) => {
          const submissionId = lastSubmissionInfo?._id;
          const initialScore =
            lastSubmissionInfo?.individualScores?.[member] || 0;
          if (!initialFormData[submissionId]) {
            initialFormData[submissionId] = {};
          }
          initialFormData[submissionId][member] = initialScore;
        });
    }
    setFormData(initialFormData);
  }, [
    assignment?.type,
    lastSubmissionInfo?._id,
    lastSubmissionInfo?.individualScores,
    studentId,
    userGroupInfo?.lastSubmissionInfo?._id,
    userGroupInfo?.lastSubmissionInfo?.individualScores,
    userGroupInfo?.members,
  ]);

  const fetchQuestionsInfo = async () => {
    try {
      if (!topicId) return;
      const res = await fetch(`/api/question/get-questions/${topicId}`);
      const data = await res.json();
      if (res.ok) {
        setQuestionsInfo(data.questions);
      }
    } catch (error) {
      console.log(error.message);
    }
  };

  useEffect(() => {
    fetchQuestionsInfo();
  }, [topicId]);

  const handleQuestionSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setAddQuestionError(null);
    setAddQuestionSuccess(null);
    try {
      const res = await fetch("/api/question/create", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          content: questionFormData,
          userId: currentUser?._id,
          topicId: topicId,
        }),
      });
      const data = await res.json();
      if (res.ok) {
        setAddQuestionSuccess("Question added successfully!");
        setQuestionFormData("");
        fetchQuestionsInfo();
      } else {
        setAddQuestionError(data.message);
      }
      setLoading(false);
    } catch (error) {
      setAddQuestionError(error.message);
      setLoading(false);
    }
  };

  const handleReplySubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setAddReplyError(null);
    setAddReplySuccess(null);
    try {
      const res = await fetch("/api/reply/create", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          content: replyFormData,
          userId: currentUser?._id,
          questionId: questionIdToReply,
        }),
      });
      const data = await res.json();
      if (res.ok) {
        setAddReplySuccess("Reply added successfully!");
        setReplyFormData("");
        fetchQuestionsInfo();
      } else {
        setAddReplyError(data.message);
      }
      setLoading(false);
    } catch (error) {
      setAddReplyError(error.message);
      setLoading(false);
    }
  };

  const handleChangeIndividual = (e, value, j) => {
    setFormData((prevFormData) => {
      const updatedScores = [...(prevFormData[e.target.id] || [])];
      updatedScores[j] = value;
      return {
        ...prevFormData,
        [e.target.id]: updatedScores,
      };
    });
  };

  const handleChangeGroup = (e, value, j, memberId) => {
    setFormData((prevFormData) => {
      const submissionId = e.target.id;
      const prevSubmission = prevFormData[submissionId] || {};
      let prevStudentScores = prevSubmission[memberId];
      prevStudentScores[j] = value;
      return {
        ...prevFormData,
        [submissionId]: {
          ...prevSubmission,
          [memberId]: prevStudentScores,
        },
      };
    });
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
        setUpdateSuccess(
          "Update overall score and feedback for student(s) successfully!"
        );
      }
      setLoading(false);
    } catch (error) {
      setUpdateError(error.message);
      setLoading(false);
    }
  };

  const handleGenerateAIScore = async (e) => {
    e.preventDefault();
    setShowModalGenerateAIScore(false);
    setIsGenerating(true);
    setGenerateError(null);
    setGenerateSuccess(null);
    try {
      const res = await fetch(
        `https://mern-lms-saxg.onrender.com/api/assignment/save-result/${classId}/${assignment?._id}/${studentId}`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
        }
      );
      const data = await res.json();
      if (!res.ok) {
        setGenerateError(data.message);
      } else {
        setGenerateSuccess("Generate AI score successfully!");
        fetchLastSubmissionInfo();
      }
      setIsGenerating(false);
    } catch (error) {
      setGenerateError(error.message);
      setIsGenerating(false);
    }
  };

  const handleApproveAISuggestion = async () => {
    setShowModalApproveAISuggestion(false);
    try {
      setApproveError(null);
      setApproveSuccess(null);
      const res = await fetch(`/api/assignment/approve/${assignmentId}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
      });
      const data = await res.json();
      if (!res.ok) {
        setApproveError(data.message);
      } else {
        setApproveSuccess("Approve AI suggestion successfully!");
        fetchLastSubmissionInfo();
      }
    } catch (error) {
      setApproveError(error.message);
    }
  };

  useEffect(() => {
    const urlParams = new URLSearchParams(location.search);
    const tabFromUrl = urlParams.get("assignmentId");
    if (tabFromUrl) {
      setAssignmentId(tabFromUrl);
    }
    const fetchMaterials = async () => {
      if (assignmentId.length === 0) {
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
  }, [location.search, assignmentId, classId]);

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
        let res;
        if (currentUser?.isEducator && studentId) {
          res = await fetch(
            `/api/submission/get-user-submissions/${assignmentId}/${studentId}`
          );
        } else if (currentUser?.isStudent && currentUser?._id) {
          res = await fetch(
            `/api/submission/get-user-submissions/${assignmentId}/${currentUser?._id}`
          );
        }
        const data = await res.json();
        if (res.ok) {
          setSubmissionsInfo(data);
        }
      } catch (error) {
        console.log(error.message);
      }
    };
    fetchSubmissionsInfo();
  }, [
    currentUser?.isEducator,
    currentUser?.isStudent,
    studentId,
    assignmentId,
    currentUser?._id,
  ]);

  const fetchLastSubmissionInfo = async () => {
    if (!classId || !assignmentId) {
      return;
    }
    try {
      let res;
      if (currentUser?.isEducator) {
        if (studentId) {
          res = await fetch(
            `/api/assignment/get-last-submission-user/${classId}/${studentId}?assignmentId=${assignmentId}`
          );
        }
      } else if (currentUser?.isStudent) {
        if (currentUser?._id) {
          res = await fetch(
            `/api/assignment/get-last-submission-user/${classId}/${currentUser?._id}?assignmentId=${assignmentId}`
          );
        }
      }
      const data = await res.json();
      if (res.ok) {
        setLastSubmissionInfo(data[0]?.lastSubmission);
        setTopicId(data[0]?.lastSubmission?.reviewRequest);
        setUserGroupInfo(data[0]?.userGroup);
      }
    } catch (error) {
      console.log(error.message);
    }
  };

  useEffect(() => {
    fetchLastSubmissionInfo();
  }, [
    currentUser?.isEducator,
    currentUser?.isStudent,
    studentId,
    assignmentId,
    classId,
    currentUser?._id,
  ]);

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
                <div className="flex flex-col gap-2 my-5 lg:w-11/12 mx-auto">
                  <div className="flex gap-2">
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
              </div>
            </Accordion>
            {moment().isBefore(assignment?.endDate) ? (
              currentUser?.isStudent && (
                <div>
                  <Link
                    to={`/class/${classId}/add-submission?assignmentId=${assignment._id}`}
                  >
                    <Button
                      variant="contained"
                      component="label"
                      style={{
                        backgroundColor: "#26597C",
                        textTransform: "none",
                      }}
                      size="large"
                    >
                      Add Submission
                    </Button>
                  </Link>
                </div>
              )
            ) : (
              <div className="font-bold text-xl text-red-600">
                <div className="w-1/4">
                  <Alert severity="error" className="border border-red-600">
                    Assignment has ended
                  </Alert>
                </div>
              </div>
            )}
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
                <div className="flex gap-2">
                  <SchoolIcon />
                  <Typography
                    component="span"
                    style={{ fontSize: "18px", fontWeight: "bold" }}
                  >
                    Grade
                  </Typography>
                </div>
              </AccordionSummary>
              <div className="flex flex-col gap-5 my-5">
                <div className="bg-white border-2 border-gray-300 shadow-md p-6 rounded-xl lg:w-11/12 mx-auto">
                  <div className="text-3xl font-bold text-[#26597C] text-center pb-6">
                    Final Score
                  </div>
                  {currentUser?.isStudent ? (
                    <>
                      <div className="flex gap-4">
                        <div className="font-bold text-xl">Overall:</div>
                        {assignment?.type === "Individual" && (
                          <div className="font-bold text-red-600 text-xl">
                            {lastSubmissionInfo?.individualScores?.length > 1
                              ? lastSubmissionInfo?.individualScores
                                  ?.slice(0, -1)
                                  ?.reduce(
                                    (sum, score, idx) =>
                                      sum +
                                      score *
                                        (lastSubmissionInfo?.coefficients?.[
                                          idx
                                        ] || 1),
                                    0
                                  )
                              : lastSubmissionInfo?.overallScore}
                          </div>
                        )}
                        {assignment?.type === "Group" && (
                          <div className="font-bold text-red-600 text-xl">
                            {lastSubmissionInfo?.individualScores?.[
                              currentUser?._id
                            ]?.length > 1
                              ? lastSubmissionInfo?.individualScores?.[
                                  currentUser?._id
                                ]
                                  ?.slice(0, -1)
                                  ?.reduce(
                                    (sum, score, idx) =>
                                      sum +
                                      score *
                                        (lastSubmissionInfo?.coefficients?.[
                                          idx
                                        ] || 1),
                                    0
                                  )
                              : lastSubmissionInfo?.overallScore}
                          </div>
                        )}
                      </div>
                      <div className="flex gap-4 items-center">
                        <div className="flex flex-col gap-2">
                          <div className="">
                            {assignment?.type === "Individual" &&
                              lastSubmissionInfo?.scoreComponent?.length >
                                0 && (
                                <div className="flex flex-col gap-2">
                                  {lastSubmissionInfo?.scoreComponent.map(
                                    (e, index) => (
                                      <div key={index} className="text-lg">
                                        (Coef: x
                                        {lastSubmissionInfo?.coefficients?.[
                                          index
                                        ] || 1}
                                        ) {e}:
                                      </div>
                                    )
                                  )}
                                </div>
                              )}
                          </div>
                          <div className="">
                            {assignment?.type === "Group" &&
                              lastSubmissionInfo?.scoreComponent?.length >
                                0 && (
                                <div className="flex flex-col gap-2">
                                  {lastSubmissionInfo?.scoreComponent.map(
                                    (e, index) => (
                                      <div key={index} className="text-lg">
                                        (Coef: x
                                        {lastSubmissionInfo?.coefficients?.[
                                          index
                                        ] || 1}
                                        ) {e}:
                                      </div>
                                    )
                                  )}
                                </div>
                              )}
                          </div>
                        </div>
                        <div className="flex flex-col gap-2">
                          <div className="">
                            {assignment?.type === "Individual" &&
                              lastSubmissionInfo?.scoreComponent?.length >
                                0 && (
                                <div className="flex flex-col gap-2">
                                  {lastSubmissionInfo?.scoreComponent.map(
                                    (_, index) => (
                                      <div
                                        key={index}
                                        className="font-semibold text-lg"
                                      >
                                        {
                                          lastSubmissionInfo
                                            ?.individualScores?.[index]
                                        }
                                      </div>
                                    )
                                  )}
                                </div>
                              )}
                          </div>
                          <div className="">
                            {assignment?.type === "Group" &&
                              lastSubmissionInfo?.scoreComponent?.length >
                                0 && (
                                <div className="flex flex-col gap-2">
                                  {lastSubmissionInfo?.scoreComponent.map(
                                    (score, index) => (
                                      <div
                                        key={index}
                                        className="font-semibold text-lg"
                                      >
                                        {
                                          lastSubmissionInfo
                                            ?.individualScores?.[
                                            currentUser._id
                                          ]?.[index]
                                        }
                                      </div>
                                    )
                                  )}
                                </div>
                              )}
                          </div>
                        </div>
                      </div>
                    </>
                  ) : (
                    <>
                      <div className="flex gap-4">
                        <div className="font-bold text-xl">Overall:</div>
                        {assignment?.type === "Individual" && (
                          <div className="font-bold text-red-600 text-xl">
                            {formData?.[lastSubmissionInfo?._id]?.length > 1
                              ? formData?.[lastSubmissionInfo?._id]
                                  ?.slice(0, -1)
                                  ?.reduce(
                                    (sum, score, idx) =>
                                      sum +
                                      score *
                                        (lastSubmissionInfo?.coefficients?.[
                                          idx
                                        ] || 1),
                                    0
                                  )
                              : lastSubmissionInfo?.overallScore}
                          </div>
                        )}
                        {assignment?.type === "Group" && (
                          <div className="font-bold text-red-600 text-xl">
                            {formData?.[lastSubmissionInfo?._id]?.[studentId]
                              ?.length > 1
                              ? formData?.[lastSubmissionInfo?._id]?.[studentId]
                                  ?.slice(0, -1)
                                  ?.reduce(
                                    (sum, score, idx) =>
                                      sum +
                                      score *
                                        (lastSubmissionInfo?.coefficients?.[
                                          idx
                                        ] || 1),
                                    0
                                  )
                              : lastSubmissionInfo?.overallScore}
                          </div>
                        )}
                      </div>
                      <div className="flex gap-4 items-center">
                        <div className="flex flex-col gap-2">
                          <div className="">
                            {assignment?.type === "Individual" &&
                              lastSubmissionInfo?.scoreComponent?.length >
                                0 && (
                                <div className="flex flex-col gap-2">
                                  {lastSubmissionInfo?.scoreComponent.map(
                                    (e, index) => (
                                      <div key={index} className="text-lg">
                                        (Coef: x
                                        {lastSubmissionInfo?.coefficients?.[
                                          index
                                        ] || 1}
                                        ) {e}:
                                      </div>
                                    )
                                  )}
                                </div>
                              )}
                          </div>
                          <div className="">
                            {assignment?.type === "Group" &&
                              lastSubmissionInfo?.scoreComponent?.length >
                                0 && (
                                <div className="flex flex-col gap-2">
                                  {lastSubmissionInfo?.scoreComponent.map(
                                    (e, index) => (
                                      <div key={index} className="text-lg">
                                        (Coef: x
                                        {lastSubmissionInfo?.coefficients?.[
                                          index
                                        ] || 1}
                                        ) {e}:
                                      </div>
                                    )
                                  )}
                                </div>
                              )}
                          </div>
                        </div>
                        <div className="flex flex-col gap-2">
                          <div className="">
                            {assignment?.type === "Individual" &&
                              lastSubmissionInfo?.scoreComponent?.length >
                                0 && (
                                <div className="flex flex-col gap-2">
                                  {lastSubmissionInfo?.scoreComponent.map(
                                    (_, index) => (
                                      <div
                                        key={index}
                                        className="font-bold text-lg"
                                      >
                                        {isEditMode ? (
                                          <div className="w-1/4">
                                            <TextInput
                                              required
                                              sizing="sm"
                                              id={lastSubmissionInfo?._id}
                                              value={
                                                lastSubmissionInfo
                                                  ?.individualScores?.[index]
                                              }
                                              onChange={(e) =>
                                                handleChangeIndividual(
                                                  e,
                                                  Number(e.target.value),
                                                  index
                                                )
                                              }
                                            />
                                          </div>
                                        ) : (
                                          <div className="font-semibold">
                                            {
                                              lastSubmissionInfo
                                                ?.individualScores?.[index]
                                            }
                                          </div>
                                        )}
                                      </div>
                                    )
                                  )}
                                </div>
                              )}
                          </div>
                          <div className="">
                            {assignment?.type === "Group" &&
                              lastSubmissionInfo?.scoreComponent?.length >
                                0 && (
                                <div className="flex flex-col gap-2">
                                  {lastSubmissionInfo?.scoreComponent.map(
                                    (_, index) => (
                                      <div
                                        key={index}
                                        className="font-bold text-lg"
                                      >
                                        {isEditMode ? (
                                          <div className="w-1/4">
                                            <TextInput
                                              required
                                              sizing="sm"
                                              id={lastSubmissionInfo?._id}
                                              value={
                                                lastSubmissionInfo
                                                  ?.individualScores?.[
                                                  studentId
                                                ]?.[index]
                                              }
                                              onChange={(e) =>
                                                handleChangeGroup(
                                                  e,
                                                  Number(e.target.value),
                                                  index,
                                                  studentId
                                                )
                                              }
                                            />
                                          </div>
                                        ) : (
                                          <div className="font-semibold">
                                            {
                                              lastSubmissionInfo
                                                ?.individualScores?.[
                                                studentId
                                              ]?.[index]
                                            }
                                          </div>
                                        )}
                                      </div>
                                    )
                                  )}
                                </div>
                              )}
                          </div>
                        </div>
                      </div>
                    </>
                  )}
                  <div className="flex flex-col gap-2 py-4">
                    <div className="font-bold text-lg">Feedback:</div>
                    {currentUser?.isStudent &&
                      assignment?.type === "Individual" &&
                      lastSubmissionInfo?.individualScores?.length >= 1 && (
                        <div className="text-gray-950">
                          {lastSubmissionInfo?.individualScores?.[
                            lastSubmissionInfo?.individualScores?.length - 1
                          ]
                            ?.split("\n")
                            .map((line, index) => (
                              <div key={index}>
                                {line?.length !== 0 && index !== 0 && <br />}
                                <p>{line}</p>
                              </div>
                            ))}
                        </div>
                      )}
                    {currentUser?.isEducator &&
                      assignment?.type === "Individual" &&
                      formData?.[lastSubmissionInfo?._id]?.length >= 1 &&
                      (isEditMode ? (
                        <div className="">
                          <Textarea
                            id={lastSubmissionInfo?._id}
                            rows="6"
                            value={
                              formData?.[lastSubmissionInfo?._id]?.[
                                formData?.[lastSubmissionInfo?._id]?.length - 1
                              ]
                            }
                            onChange={(e) =>
                              handleChangeIndividual(
                                e,
                                e.target.value,
                                formData?.[lastSubmissionInfo?._id]?.length - 1
                              )
                            }
                          />
                        </div>
                      ) : (
                        lastSubmissionInfo?.individualScores?.length >= 1 && (
                          <div className="text-gray-950">
                            {lastSubmissionInfo?.individualScores?.[
                              lastSubmissionInfo?.individualScores?.length - 1
                            ]
                              ?.split("\n")
                              .map((line, index) => (
                                <div key={index}>
                                  {line?.length !== 0 && index !== 0 && <br />}
                                  <p>{line}</p>
                                </div>
                              ))}
                          </div>
                        )
                      ))}
                    {currentUser?.isStudent && assignment?.type === "Group" && (
                      <div className="text-gray-950">
                        {
                          lastSubmissionInfo?.individualScores?.[
                            currentUser._id
                          ]?.[
                            lastSubmissionInfo?.individualScores?.[
                              currentUser._id
                            ]?.length - 1
                          ]
                        }
                      </div>
                    )}
                    {currentUser?.isEducator &&
                      assignment?.type === "Group" &&
                      formData?.[lastSubmissionInfo?._id]?.[studentId]
                        ?.length >= 1 &&
                      (isEditMode ? (
                        <div className="">
                          <Textarea
                            id={lastSubmissionInfo?._id}
                            rows="6"
                            value={
                              formData?.[lastSubmissionInfo?._id]?.[
                                studentId
                              ]?.[
                                formData?.[lastSubmissionInfo?._id]?.[studentId]
                                  ?.length - 1
                              ]
                            }
                            onChange={(e) =>
                              handleChangeGroup(
                                e,
                                e.target.value,
                                formData?.[lastSubmissionInfo?._id]?.[studentId]
                                  ?.length - 1,
                                studentId
                              )
                            }
                          />
                        </div>
                      ) : (
                        lastSubmissionInfo?.individualScores?.[studentId]
                          ?.length >= 1 && (
                          <div className="text-gray-950">
                            {lastSubmissionInfo?.individualScores?.[
                              studentId
                            ]?.[
                              lastSubmissionInfo?.individualScores?.[studentId]
                                ?.length - 1
                            ]
                              ?.split("\n")
                              .map((line, index) => (
                                <div key={index}>
                                  {line?.length !== 0 && index !== 0 && <br />}
                                  <p>{line}</p>
                                </div>
                              ))}
                          </div>
                        )
                      ))}
                  </div>
                  {currentUser?.isEducator && isEditMode && (
                    <div className="pb-4 flex justify-end">
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
                  )}
                  {updateError && (
                    <Alert severity="error" className="border border-red-600">
                      {updateError}
                    </Alert>
                  )}
                  {updateSuccess && (
                    <Alert
                      severity="success"
                      className="border border-green-600"
                    >
                      {updateSuccess}
                    </Alert>
                  )}
                </div>

                {currentUser?.isEducator && (
                  <div className="bg-white border-2 border-gray-300 shadow-md p-6 rounded-xl lg:w-11/12 mx-auto">
                    <div className="text-3xl font-bold text-[#26597C] text-center pb-6">
                      AI Suggestion Score
                    </div>
                    <div className="flex gap-4 pb-2">
                      <div className="font-bold text-xl">Overall:</div>
                      <div className="font-bold text-red-600 text-xl">
                        {lastSubmissionInfo?.score?.length > 0
                          ? lastSubmissionInfo?.score?.reduce(
                              (sum, score, idx) =>
                                sum +
                                score *
                                  (lastSubmissionInfo?.coefficients?.[idx] ||
                                    1),
                              0
                            )
                          : lastSubmissionInfo?.overallAIScore}
                      </div>
                    </div>
                    <div className="flex gap-4 items-center">
                      {lastSubmissionInfo?.scoreComponent?.length > 0 && (
                        <div className="flex flex-col gap-2">
                          {lastSubmissionInfo?.scoreComponent.map(
                            (e, index) => (
                              <div key={index} className="text-lg">
                                (Coef: x
                                {lastSubmissionInfo?.coefficients?.[index] || 1}
                                ) {e}:
                              </div>
                            )
                          )}
                        </div>
                      )}
                      {lastSubmissionInfo?.scoreComponent?.length > 0 && (
                        <div className="flex flex-col gap-2">
                          {lastSubmissionInfo?.scoreComponent.map(
                            (_, index) => (
                              <div
                                key={index}
                                className="font-semibold text-lg"
                              >
                                {lastSubmissionInfo?.score?.[index]}
                              </div>
                            )
                          )}
                        </div>
                      )}
                    </div>
                    {currentUser?.isEducator && (
                      <div className="flex flex-col gap-2 py-4">
                        <div className="font-bold text-lg">Feedback: </div>
                        <div className="text-gray-950">
                          {lastSubmissionInfo?.feedback?.length > 0 &&
                            lastSubmissionInfo?.feedback
                              ?.split("\n")
                              .map((line, index) => (
                                <div key={index}>
                                  {line?.length !== 0 && index !== 0 && <br />}
                                  <p>{line}</p>
                                </div>
                              ))}
                        </div>
                      </div>
                    )}
                    <div className="flex gap-2 justify-end pb-4">
                      {currentUser?.isEducator && isEditMode && (
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
                            setShowModalGenerateAIScore(true);
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
                      )}
                      {currentUser?.isEducator && isEditMode && (
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
                        >
                          Approve AI score
                        </Button>
                      )}
                    </div>
                    {generateError && (
                      <Alert severity="error" className="border border-red-600">
                        {generateError}
                      </Alert>
                    )}
                    {generateSuccess && (
                      <Alert
                        severity="success"
                        className="border border-green-600"
                      >
                        {generateSuccess}
                      </Alert>
                    )}
                    {approveError && (
                      <Alert severity="error" className="border border-red-600">
                        {approveError}
                      </Alert>
                    )}
                    {approveSuccess && (
                      <Alert
                        severity="success"
                        className="border border-green-600"
                      >
                        {approveSuccess}
                      </Alert>
                    )}
                  </div>
                )}

                <div className="bg-white border-2 border-gray-300 shadow-md p-6 rounded-xl lg:w-11/12 mx-auto">
                  <div className="text-3xl font-bold text-[#26597C] text-center">
                    Review Request
                  </div>
                  <div className="text-center my-6">
                    <button
                      type="button"
                      onClick={() => setIsShowAddQuestion(!isShowAddQuestion)}
                      className="inline-flex items-center gap-2 text-cyan-700 hover:underline font-semibold text-lg"
                    >
                      {isShowAddQuestion ? (
                        <>
                          <i className="fa-solid fa-xmark"></i> Cancel
                        </>
                      ) : (
                        <>
                          <i className="fa-solid fa-plus"></i> Add Question
                        </>
                      )}
                    </button>
                  </div>
                  {isShowAddQuestion && (
                    <form
                      onSubmit={handleQuestionSubmit}
                      className="mt-6 bg-gray-100 border-2 border-gray-300 p-6 rounded-xl shadow-md"
                    >
                      <Textarea
                        placeholder="Write your question here..."
                        rows="4"
                        maxLength="200"
                        onChange={(e) => setQuestionFormData(e.target.value)}
                        value={questionFormData}
                        className="mb-3"
                      />
                      <div className="text-right text-sm text-gray-500 mb-2">
                        {questionFormData?.length}/200 characters
                      </div>
                      <div className="flex justify-end gap-2">
                        <Button
                          variant="contained"
                          type="submit"
                          style={{
                            backgroundColor: "#26597C",
                            color: "#ffffff",
                            textTransform: "none",
                          }}
                          disabled={loading}
                        >
                          Submit Question
                        </Button>
                        <Button
                          variant="contained"
                          style={{
                            backgroundColor: "oklch(0.577 0.245 27.325)",
                            color: "#ffffff",
                            textTransform: "none",
                          }}
                          onClick={() =>
                            setIsShowAddQuestion(!isShowAddQuestion)
                          }
                        >
                          Cancel
                        </Button>
                      </div>
                      {addQuestionError && (
                        <Alert
                          severity="error"
                          className="mt-4 border border-red-600"
                        >
                          {addQuestionError}
                        </Alert>
                      )}
                      {addQuestionSuccess && (
                        <Alert
                          severity="success"
                          className="mt-4 border border-green-600"
                        >
                          {addQuestionSuccess}
                        </Alert>
                      )}
                    </form>
                  )}
                  {questionsInfo?.length > 0 &&
                    questionsInfo.map((question, i) => (
                      <div
                        className="bg-gray-100 p-4 mt-4 rounded-xl border-2 border-gray-300 shadow-md"
                        key={i}
                      >
                        <div className="flex items-center gap-4">
                          <Avatar
                            img={question?.userId?.profilePicture}
                            rounded
                          />
                          <div className="text-gray-600 flex gap-2">
                            <span className="font-bold">
                              {question?.userId?.name}
                            </span>
                            <span>-</span>
                            <span>
                              {question?.createdAt
                                ? moment(question?.createdAt).format(
                                    "HH:mm:ss DD/MM/YYYY"
                                  )
                                : "---"}
                            </span>
                          </div>
                        </div>
                        <p className="text-lg text-gray-950 mb-2 ml-14 line-clamp-2">
                          {question?.content}
                        </p>
                        <div
                          className="text-cyan-700 hover:underline font-semibold text-right mr-2"
                          onClick={() => setQuestionIdToReply(question._id)}
                        >
                          Reply
                        </div>
                        {questionIdToReply === question?._id && (
                          <form
                            onSubmit={handleReplySubmit}
                            className="mt-4 ml-12 bg-sky-100 border border-cyan-600 p-4 rounded-xl shadow-md"
                          >
                            <Textarea
                              placeholder="Write your reply here..."
                              rows="3"
                              maxLength="200"
                              onChange={(e) => setReplyFormData(e.target.value)}
                              value={replyFormData}
                              className="mb-3"
                            />
                            <div className="text-right text-sm text-gray-500 mb-2">
                              {replyFormData?.length}/200 characters
                            </div>
                            <div className="flex justify-end gap-2">
                              <Button
                                variant="contained"
                                type="submit"
                                style={{
                                  backgroundColor: "#26597C",
                                  color: "#ffffff",
                                  textTransform: "none",
                                }}
                                disabled={loading}
                              >
                                Submit Reply
                              </Button>
                              <Button
                                variant="contained"
                                onClick={() => {
                                  setQuestionIdToReply(null);
                                  setReplyFormData("");
                                  setAddReplyError(null);
                                  setAddReplySuccess(null);
                                }}
                                style={{
                                  backgroundColor: "oklch(0.577 0.245 27.325)",
                                  color: "#ffffff",
                                  textTransform: "none",
                                }}
                              >
                                Cancel
                              </Button>
                            </div>
                            {addReplyError && (
                              <Alert
                                severity="error"
                                className="mt-4 border border-red-600"
                              >
                                {addReplyError}
                              </Alert>
                            )}
                            {addReplySuccess && (
                              <Alert
                                severity="success"
                                className="mt-4 border border-green-600"
                              >
                                {addReplySuccess}
                              </Alert>
                            )}
                          </form>
                        )}
                        {question?.replies?.length > 0 &&
                          question?.replies.map((reply, i) => (
                            <div
                              key={i}
                              className="ml-12 mt-4 border border-cyan-600 pl-4 py-2 bg-sky-100 rounded-xl shadow-md"
                            >
                              <div className="flex items-center gap-4">
                                <Avatar
                                  img={reply?.userId?.profilePicture}
                                  // size="sm"
                                  rounded
                                />
                                <div className="text-gray-600 flex gap-2">
                                  <span className="font-bold">
                                    {reply?.userId?.name}
                                  </span>
                                  <span>-</span>
                                  <span>
                                    {reply?.createdAt
                                      ? moment(reply?.createdAt).format(
                                          "HH:mm:ss DD/MM/YYYY"
                                        )
                                      : "---"}
                                  </span>
                                </div>
                              </div>
                              <div className="text-lg text-gray-950 ml-14">
                                {reply?.content}
                              </div>
                            </div>
                          ))}
                      </div>
                    ))}
                </div>
              </div>
            </Accordion>
            {submissionsInfo?.length === 0 ? (
              <div className="w-1/5">
                <Alert severity="error" className="mb-4 border border-red-600">
                  No submission yet
                </Alert>
              </div>
            ) : (
              <div className="font-bold text-3xl text-[#26597C] text-center">
                Submission Attempts
              </div>
            )}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 my-4">
              {submissionsInfo?.length > 0 &&
                submissionsInfo.map((submission, index) => (
                  <div className="w-full" key={index}>
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
                        <div className="flex items-center justify-between w-full">
                          <div className="flex gap-2 items-center">
                            <FolderIcon />
                            <Typography
                              component="span"
                              style={{ fontSize: "18px", fontWeight: "bold" }}
                            >
                              Attempts {index + 1}
                            </Typography>
                          </div>
                        </div>
                      </AccordionSummary>
                      <div className="flex gap-2 my-5 lg:w-11/12 mx-auto">
                        <div className="flex flex-col gap-2">
                          <div className="font-bold">Submission Date: </div>
                          <div className="font-bold">Description: </div>
                          <div className="font-bold">Attachment: </div>
                        </div>
                        <div className="flex flex-col gap-2">
                          <div>
                            {submission?.updatedAt
                              ? moment(submission?.updatedAt).format(
                                  "HH:mm:ss DD/MM/YYYY"
                                )
                              : "---"}
                          </div>
                          <div>{submission?.description || "---"}</div>
                          <div className="flex flex-col gap-2">
                            {submission?.submissionUrls.map((file, index) => (
                              <div className="flex items-center" key={index}>
                                <img
                                  src={pdf}
                                  alt="pdf icon"
                                  className="w-6 h-6"
                                />
                                <Link
                                  to={file}
                                  underline="hover"
                                  target="_blank"
                                  className="hover:underline text-cyan-600 ml-1"
                                >
                                  {submission?.nameFiles[index]}
                                </Link>
                              </div>
                            ))}
                          </div>
                        </div>
                      </div>
                    </Accordion>
                  </div>
                ))}
            </div>
          </div>
        </div>
        <div className="basis-3/12 ml-5">
          <ClassSidebar classId={classId} />
        </div>
      </div>
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
        show={showModalGenerateAIScore}
        onClose={() => setShowModalGenerateAIScore(false)}
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
                onClick={handleGenerateAIScore}
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
                onClick={() => setShowModalGenerateAIScore(false)}
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

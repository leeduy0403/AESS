import { useEffect, useState } from "react";
import { Link, useLocation, useParams } from "react-router-dom";
import ClassSidebar from "../components/ClassSidebar";
import { useDispatch, useSelector } from "react-redux";
import { toggleIsEditMode } from "../redux/isEditMode/isEditModeSlice";
import moment from "moment";
import { Avatar, Textarea } from "flowbite-react";
import { Alert, Button } from "@mui/material";
import { ArrowBack as ArrowBackIcon } from "@mui/icons-material";

export default function ViewQuestions() {
  const dispatch = useDispatch();
  const { currentUser } = useSelector((state) => state.user);
  const { classId } = useParams();
  const { isEditMode } = useSelector((state) => state.isEditMode);
  const { tabIndex } = useSelector((state) => state.tabIndex);
  const location = useLocation();
  const [tabValue, setTabValue] = useState("");
  const [classInfo, setClassInfo] = useState({});
  const [topicId, setTopicId] = useState(null);
  const [questionsInfo, setQuestionsInfo] = useState([]);
  const [topicInfo, setTopicInfo] = useState("");
  const [questionFormData, setQuestionFormData] = useState("");
  const [replyFormData, setReplyFormData] = useState("");
  const [questionIdToReply, setQuestionIdToReply] = useState(null);
  const [isShowAddQuestion, setIsShowAddQuestion] = useState(false);
  const [addQuestionError, setAddQuestionError] = useState(null);
  const [addReplyError, setAddReplyError] = useState(null);
  const [addQuestionSuccess, setAddQuestionSuccess] = useState(null);
  const [addReplySuccess, setAddReplySuccess] = useState(null);
  // console.log(questionsInfo);
  // console.log(topicInfo);
  console.log(questionFormData);
  console.log(questionIdToReply);
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
    const topicIdFromUrl = urlParams.get("topicId");
    if (topicIdFromUrl) {
      setTopicId(topicIdFromUrl);
    }
  }, [location.search]);

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

  const fetchQuestionsInfo = async () => {
    try {
      if (!topicId) return;
      const res = await fetch(`/api/question/get-questions/${topicId}`);
      const data = await res.json();
      if (res.ok) {
        setQuestionsInfo(data.questions);
        setTopicInfo(data.topicInfo);
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
          userId: currentUser._id,
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
    } catch (error) {
      setAddQuestionError(error.message);
    }
  };

  const handleReplySubmit = async (e) => {
    e.preventDefault();
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
          userId: currentUser._id,
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
    } catch (error) {
      setAddReplyError(error.message);
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
        <div className="basis-9/12 flex flex-col gap-6">
          <Link to={`/class/${classId}`}>
            <div className="flex gap-2 items-center pt-1 hover:underline text-cyan-600 font-semibold">
              <ArrowBackIcon />
              <span>Back to {tabValue} tab</span>
            </div>
          </Link>
          <div className="bg-white border-2 border-gray-300 shadow-md p-6 rounded-xl">
            <h3 className="text-3xl font-semibold text-[#26597C] text-center mb-4">
              {topicInfo?.content}
            </h3>
            <div className="flex flex-wrap gap-6 justify-center  text-gray-600 border-b-2 border-gray-300 pb-4">
              <div className="flex items-center gap-2">
                <span className="font-semibold">Created:</span>
                {topicInfo?.createdAt
                  ? moment(topicInfo?.createdAt).format("HH:mm:ss DD/MM/YYYY")
                  : "---"}
              </div>
              <div className="flex items-center gap-2">
                <span className="font-semibold">By:</span>
                <Avatar
                  img={topicInfo?.userId?.profilePicture}
                  // size="sm"
                  rounded
                />
                <span className="text-cyan-600 hover:underline">
                  {topicInfo?.userId?.name}
                </span>
              </div>
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
                    onClick={() => setIsShowAddQuestion(!isShowAddQuestion)}
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
                    <Avatar img={question?.userId?.profilePicture} rounded />
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
        <div className="basis-3/12 ml-5 sticky top-4 h-fit">
          <ClassSidebar classId={classId} />
        </div>
      </div>
    </div>
  );
}

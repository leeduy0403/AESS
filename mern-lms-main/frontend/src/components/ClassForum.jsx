import Accordion from "@mui/material/Accordion";
import AccordionSummary from "@mui/material/AccordionSummary";
import Typography from "@mui/material/Typography";
import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import moment from "moment";
import {
  ExpandMore as ExpandMoreIcon,
  Forum as ForumIcon,
} from "@mui/icons-material";
import { Textarea } from "flowbite-react";
import { Alert, Button } from "@mui/material";
import { useSelector } from "react-redux";

export default function ClassForum() {
  const { classId } = useParams();
  const { currentUser } = useSelector((state) => state.user);
  const { isEditMode } = useSelector((state) => state.isEditMode);

  const [forumsInfo, setForumsInfo] = useState([]);
  const [forumFormData, setForumFormData] = useState("");
  const [isShowAddForum, setIsShowAddForum] = useState(false);
  const [topicFormDataByForum, setTopicFormDataByForum] = useState({});
  const [isShowAddTopicByForum, setIsShowAddTopicByForum] = useState({});
  const [addForumError, setAddForumError] = useState(null);
  const [addForumSuccess, setAddForumSuccess] = useState(null);
  const [addTopicError, setAddTopicError] = useState(null);
  const [addTopicSuccess, setAddTopicSuccess] = useState(null);

  const fetchForumsInfo = async () => {
    try {
      const res = await fetch(`/api/forum/get-forums/${classId}`);
      const data = await res.json();
      if (res.ok) {
        setForumsInfo(data);
      }
    } catch (error) {
      console.log(error.message);
    }
  };

  useEffect(() => {
    fetchForumsInfo();
  }, [classId]);

  const handleForumSubmit = async (e) => {
    e.preventDefault();
    setAddForumError(null);
    setAddForumSuccess(null);
    try {
      const res = await fetch("/api/forum/create", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          title: forumFormData,
          classId: classId,
        }),
      });
      const data = await res.json();
      if (res.ok) {
        setAddForumSuccess("Forum added successfully!");
        setForumFormData("");
        fetchForumsInfo(); // Refresh forum list
      } else {
        setAddForumError(data.message);
      }
    } catch (error) {
      setAddForumError(error.message);
    }
  };

  const handleTopicSubmit = async (e, forumId) => {
    e.preventDefault();
    setAddTopicError(null);
    setAddTopicSuccess(null);
    try {
      const res = await fetch("/api/topic/create", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          content: topicFormDataByForum[forumId],
          userId: currentUser?._id,
          forumId: forumId,
        }),
      });
      const data = await res.json();
      if (res.ok) {
        setAddTopicSuccess("Topic added successfully!");
        setTopicFormDataByForum((prev) => ({ ...prev, [forumId]: "" }));
        fetchForumsInfo(); // Refresh topic list
      } else {
        setAddTopicError(data.message);
      }
    } catch (error) {
      setAddTopicError(error.message);
    }
  };

  return (
    <div className="">
      {currentUser?.isEducator && isEditMode && (
        <div className="text-center">
          <button
            type="button"
            onClick={() => setIsShowAddForum(!isShowAddForum)}
            className="inline-flex items-center gap-2 text-cyan-700 hover:underline font-semibold text-lg"
          >
            {isShowAddForum ? (
              <>
                <i className="fa-solid fa-xmark"></i> Cancel
              </>
            ) : (
              <>
                <i className="fa-solid fa-plus"></i> Add Forum
              </>
            )}
          </button>
          {isShowAddForum && (
            <form
              onSubmit={handleForumSubmit}
              className="my-6 bg-gray-100 border-2 border-gray-300 p-6 rounded-xl shadow-md"
            >
              <Textarea
                placeholder="Write your forum here..."
                rows="2"
                maxLength="200"
                onChange={(e) => setForumFormData(e.target.value)}
                value={forumFormData}
                className="mb-3"
              />
              <div className="text-right text-sm text-gray-500 mb-2">
                {forumFormData?.length}/200 characters
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
                  Submit Forum
                </Button>
                <Button
                  variant="contained"
                  style={{
                    backgroundColor: "oklch(0.577 0.245 27.325)",
                    color: "#ffffff",
                    textTransform: "none",
                  }}
                  onClick={() => setIsShowAddForum(false)}
                >
                  Cancel
                </Button>
              </div>
              {addForumError && (
                <Alert severity="error" className="mt-4 border border-red-600">
                  {addForumError}
                </Alert>
              )}
              {addForumSuccess && (
                <Alert
                  severity="success"
                  className="mt-4 border border-green-600"
                >
                  {addForumSuccess}
                </Alert>
              )}
            </form>
          )}
        </div>
      )}

      {forumsInfo?.length > 0 ? (
        forumsInfo.map((forum, index) => (
          <Accordion
            defaultExpanded
            className="mb-4"
            key={index}
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
                <ForumIcon />
                <Typography sx={{ fontSize: 18, fontWeight: "bold" }}>
                  {forum?.title}
                </Typography>
              </div>
            </AccordionSummary>
            <div className="my-5 lg:w-11/12 mx-auto px-2">
              <div className="text-center my-6">
                <button
                  type="button"
                  onClick={() =>
                    setIsShowAddTopicByForum((prev) => ({
                      ...prev,
                      [forum._id]: !prev[forum._id],
                    }))
                  }
                  className="inline-flex items-center gap-2 text-cyan-700 hover:underline font-semibold text-lg"
                >
                  {isShowAddTopicByForum[forum._id] ? (
                    <>
                      <i className="fa-solid fa-xmark"></i> Cancel
                    </>
                  ) : (
                    <>
                      <i className="fa-solid fa-plus"></i> Add Topic
                    </>
                  )}
                </button>
              </div>
              {isShowAddTopicByForum[forum._id] && (
                <form
                  onSubmit={(e) => handleTopicSubmit(e, forum._id)}
                  className="mb-6 bg-gray-100 border-2 border-gray-300 p-6 rounded-xl shadow-md"
                >
                  <Textarea
                    placeholder="Write your topic here..."
                    rows="4"
                    maxLength="200"
                    onChange={(e) =>
                      setTopicFormDataByForum((prev) => ({
                        ...prev,
                        [forum._id]: e.target.value,
                      }))
                    }
                    value={topicFormDataByForum[forum._id] || ""}
                    className="mb-3"
                  />
                  <div className="text-right text-sm text-gray-500 mb-2">
                    {topicFormDataByForum[forum._id]?.length || 0}/200
                    characters
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
                      Submit Topic
                    </Button>
                    <Button
                      variant="contained"
                      style={{
                        backgroundColor: "oklch(0.577 0.245 27.325)",
                        color: "#ffffff",
                        textTransform: "none",
                      }}
                      onClick={() =>
                        setIsShowAddTopicByForum((prev) => ({
                          ...prev,
                          [forum._id]: false,
                        }))
                      }
                    >
                      Cancel
                    </Button>
                  </div>
                  {addTopicError && (
                    <Alert
                      severity="error"
                      className="mt-4 border border-red-600"
                    >
                      {addTopicError}
                    </Alert>
                  )}
                  {addTopicSuccess && (
                    <Alert
                      severity="success"
                      className="mt-4 border border-green-600"
                    >
                      {addTopicSuccess}
                    </Alert>
                  )}
                </form>
              )}
              <div className="overflow-x-auto rounded-lg shadow-sm">
                <table className="w-full text-sm border-collapse">
                  <thead>
                    <tr className="bg-gray-200 text-gray-700 text-left">
                      <th className="p-4">Topic</th>
                      <th className="p-4">Created by</th>
                      <th className="p-4">Last updated</th>
                      <th className="p-4"># Ans</th>
                      <th className="p-4">Follow</th>
                    </tr>
                  </thead>
                  <tbody>
                    {forum?.topics?.length > 0 ? (
                      forum?.topics.map((topic, i) => (
                        <tr
                          key={i}
                          className="hover:bg-gray-50 border-t border-gray-200"
                        >
                          <td className="p-4 w-1/3">
                            <Link
                              to={`/class/${classId}/view-questions?topicId=${topic?._id}`}
                              className="hover:underline text-sky-600 font-medium line-clamp-1"
                            >
                              {topic?.content}
                            </Link>
                          </td>
                          <td className="p-4">
                            <div className="text-gray-800">
                              {topic?.userId?.name}
                            </div>
                            <div className="text-gray-500 text-sm">
                              {topic?.createdAt
                                ? moment(topic?.createdAt).format(
                                    "HH:mm:ss DD/MM/YYYY"
                                  )
                                : "---"}
                            </div>
                          </td>
                          <td className="p-4 text-gray-700">
                            {topic?.lastUpdated
                              ? moment(topic?.lastUpdated).format(
                                  "HH:mm:ss DD/MM/YYYY"
                                )
                              : "---"}
                          </td>
                          <td className="p-4">
                            <span className="bg-sky-100 text-sky-800 px-2 py-1 rounded-full text-xs font-semibold">
                              {topic?.totalReplies}
                            </span>
                          </td>
                          <td className="p-4 text-center text-gray-400">–</td>
                        </tr>
                      ))
                    ) : (
                      <tr>
                        <td
                          colSpan="5"
                          className="p-4 text-center text-gray-500"
                        >
                          No topics available in this forum.
                        </td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>
            </div>
          </Accordion>
        ))
      ) : (
        <div className="text-center text-gray-500 pt-4">
          No forum data found for this class.
        </div>
      )}
    </div>
  );
}

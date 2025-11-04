import Accordion from "@mui/material/Accordion";
import AccordionSummary from "@mui/material/AccordionSummary";
import Typography from "@mui/material/Typography";
import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import moment from "moment";
import pdf from "../assets/pdf.png";
import folder from "../assets/folder.png";
import { ExpandMore as ExpandMoreIcon } from "@mui/icons-material";
import { useSelector } from "react-redux";

export default function ClassSidebar() {
  const { classId } = useParams();
  const { currentUser } = useSelector((state) => state.user);
  const [ongoingAssignments, setOngoingAssignments] = useState([]);
  const [recentUploadMaterials, setRecentUploadMaterials] = useState([]);
  const [showMoreOngoingAssignments, setShowMoreOngoingAssignments] =
    useState(true);
  const [showMoreRecentUploadMaterials, setShowMoreRecentUploadMaterials] =
    useState(true);

  useEffect(() => {
    const fetchOngoingAssignments = async () => {
      try {
        const res = await fetch(
          `/api/assignment/get-ongoing-assignments/${classId}`
        );
        const data = await res.json();
        if (res.ok) {
          setOngoingAssignments(data);
          if (data.length < 5) {
            setShowMoreOngoingAssignments(false);
          }
        }
      } catch (error) {
        console.log(error.message);
      }
    };
    fetchOngoingAssignments();
  }, [classId]);

  const handleShowMoreOngoingAssignments = async () => {
    const startIndex = ongoingAssignments.length;
    try {
      const res = await fetch(
        `/api/assignment/get-ongoing-assignments/${classId}?startIndex=${startIndex}`
      );
      const data = await res.json();
      if (res.ok) {
        setOngoingAssignments((prev) => [...prev, ...data]);
        if (data.length < 5) {
          setShowMoreOngoingAssignments(false);
        }
      }
    } catch (error) {
      console.log(error.message);
    }
  };

  useEffect(() => {
    const fetchRecentUploadMaterials = async () => {
      try {
        const res = await fetch(`/api/material/get-materials/${classId}`);
        const data = await res.json();
        if (res.ok) {
          setRecentUploadMaterials(data);
        }
      } catch (error) {
        console.log(error.message);
      }
    };
    fetchRecentUploadMaterials();
  }, [classId]);

  const handleShowMoreRecentUploadMaterials = async () => {
    const startIndex = recentUploadMaterials.length;
    try {
      const res = await fetch(
        `/api/material/get-materials/${classId}?startIndex=${startIndex}`
      );
      const data = await res.json();
      if (res.ok) {
        setRecentUploadMaterials((prev) => [...prev, ...data]);
        if (data.length < 5) {
          setShowMoreRecentUploadMaterials(false);
        }
      }
    } catch (error) {
      console.log(error.message);
    }
  };

  return (
    <div className="mb-6">
      {/* Ongoing Assignments */}
      <Accordion defaultExpanded className="mb-4">
        <AccordionSummary
          expandIcon={<ExpandMoreIcon sx={{ color: "white" }} />}
          style={{
            backgroundColor: "#26597C",
            color: "#FFFFFF",
            borderRadius: "12px 12px 0 0",
          }}
        >
          <div className="flex gap-3 items-center">
            <i className="fa-regular fa-hourglass-half text-white"></i>
            <Typography
              component="span"
              style={{ fontSize: "17px", fontWeight: "bold" }}
            >
              Ongoing Assignments
            </Typography>
          </div>
        </AccordionSummary>

        <div className="flex flex-col gap-4 py-4 px-3">
          {ongoingAssignments?.length > 0 &&
            ongoingAssignments.map(
              (assignment, index) =>
                !assignment?.isHidden && (
                  <div
                    key={index}
                    className="bg-gray-100 hover:bg-sky-100 transition p-3 rounded-lg shadow-sm border-2 border-gray-300"
                  >
                    <div className="flex items-center gap-3 mb-1">
                      <img src={folder} alt="folder icon" className="w-5 h-5" />
                      {currentUser?.isStudent && (
                        <Link
                          to={`/class/${classId}/view-attempts?assignmentId=${assignment?._id}`}
                          className="text-sm font-medium hover:underline"
                        >
                          {assignment?.title}
                        </Link>
                      )}
                      {currentUser?.isEducator && (
                        <Link
                          to={`/class/${classId}/view-submissions?assignmentId=${assignment?._id}`}
                          className="text-sm font-medium hover:underline"
                        >
                          {assignment?.title}
                        </Link>
                      )}
                    </div>
                    <div className="text-xs text-red-600 ml-8 font-semibold">
                      Due:{" "}
                      {assignment?.endDate
                        ? moment(assignment?.endDate).format(
                            "HH:mm:ss DD/MM/YYYY"
                          )
                        : "---"}
                    </div>
                  </div>
                )
            )}

          {showMoreOngoingAssignments && (
            <button
              onClick={handleShowMoreOngoingAssignments}
              className="text-sm text-sky-600 hover:underline w-full mt-2"
            >
              Show More
            </button>
          )}
        </div>
      </Accordion>

      {/* Recent Uploads */}
      <Accordion defaultExpanded className="shadow-md rounded-xl">
        <AccordionSummary
          expandIcon={<ExpandMoreIcon sx={{ color: "white" }} />}
          style={{
            backgroundColor: "#26597C",
            color: "#FFFFFF",
            borderRadius: "12px 12px 0 0",
          }}
        >
          <div className="flex gap-3 items-center">
            <i className="fa-solid fa-clock-rotate-left text-white"></i>
            <Typography
              component="span"
              style={{ fontSize: "17px", fontWeight: "bold" }}
            >
              Recent Uploads
            </Typography>
          </div>
        </AccordionSummary>

        <div className="flex flex-col gap-4 py-4 px-3">
          {recentUploadMaterials?.length > 0 &&
            recentUploadMaterials.map(
              (material, i) =>
                !material?.isHidden &&
                material?.nameFiles?.length > 0 &&
                material?.nameFiles.map((url, j) => (
                  <div
                    key={`${i}-${j}`}
                    className="bg-gray-100 hover:bg-sky-100 transition p-3 rounded-lg shadow-sm border-2 border-gray-300"
                  >
                    <div className="flex items-center gap-3 mb-1">
                      <img src={pdf} alt="pdf icon" className="w-5 h-5" />
                      <Link
                        to={material?.materialUrls[j]}
                        target="_blank"
                        className="text-sm font-medium hover:underline"
                      >
                        {url?.replace(/\.pdf$/, "")}
                      </Link>
                    </div>
                    <div className="text-xs text-cyan-600 ml-8 font-semibold">
                      Upload:{" "}
                      {material?.createdAt
                        ? moment(material?.createdAt).format("DD/MM/YYYY")
                        : "---"}
                    </div>
                  </div>
                ))
            )}

          {showMoreRecentUploadMaterials && (
            <button
              onClick={handleShowMoreRecentUploadMaterials}
              className="text-sm text-sky-600 hover:underline w-full"
            >
              Show More
            </button>
          )}
        </div>
      </Accordion>
    </div>
  );
}

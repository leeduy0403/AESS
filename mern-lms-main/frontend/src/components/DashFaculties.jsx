import { useEffect, useState } from "react";
import { useSelector } from "react-redux";
import { Modal, Table } from "flowbite-react";
import { HiOutlineExclamationCircle } from "react-icons/hi";
import { Button } from "@mui/material";

export default function DashFaculties() {
  const { currentUser } = useSelector((state) => state.user);
  const [faculties, setFaculties] = useState([]);
  const [showMore, setShowMore] = useState(true);
  const [showModal, setShowModal] = useState(false);
  const [facultyIdToDelete, setFacultyIdToDelete] = useState("");

  useEffect(() => {
    const fetchFaculties = async () => {
      try {
        const res = await fetch("/api/faculty/get");
        const data = await res.json();
        if (res.ok) {
          setFaculties(data);
          if (data.length < 9) {
            setShowMore(false);
          }
        }
      } catch (error) {
        console.log(error.message);
      }
    };
    if (currentUser?.isAdmin) {
      fetchFaculties();
    }
  }, [currentUser?._id, currentUser?.isAdmin]);

  const handleShowMore = async () => {
    const startIndex = faculties.length;
    try {
      const res = await fetch(`/api/faculty/get?startIndex=${startIndex}`);
      const data = await res.json();
      if (res.ok) {
        setFaculties((prev) => [...prev, ...data]);
        if (data?.length < 9) {
          setShowMore(false);
        }
      }
    } catch (error) {
      console.log(error.message);
    }
  };
  const handleDeleteFaculty = async () => {
    setShowModal(false);
    try {
      const res = await fetch(`/api/faculty/delete/${facultyIdToDelete}`, {
        method: "DELETE",
      });
      const data = await res.json();
      if (!res.ok) {
        console.log(data.message);
      } else {
        setFaculties((prev) =>
          prev.filter((course) => course?._id !== facultyIdToDelete)
        );
      }
    } catch (error) {
      console.log(error.message);
    }
  };
  return (
    <div className="table-auto overflow-x-scroll md:mx-auto p-3 scrollbar scrollbar-track-slate-100 scrollbar-thumb-slate-300 dark:scrollbar-track-slate-700 dark:scrollbar-thumb-slate-500 lg:w-3/4">
      <h1 className="my-7 text-center font-semibold text-3xl">Faculties</h1>
      {currentUser?.isAdmin && faculties?.length > 0 ? (
        <>
          <Table hoverable className="shadow-md">
            <Table.Head>
              <Table.HeadCell>Date updated</Table.HeadCell>
              <Table.HeadCell>Faculty</Table.HeadCell>
              <Table.HeadCell>Delete</Table.HeadCell>
              {/* <Table.HeadCell>Edit</Table.HeadCell> */}
            </Table.Head>
            {faculties?.length > 0 &&
              faculties.map((faculty) => (
                <Table.Body className="divide-y" key={faculty?._id}>
                  <Table.Row className="bg-white dark:border-gray-700 dark:bg-gray-800">
                    <Table.Cell>
                      {new Date(faculty?.updatedAt).toLocaleDateString()}
                    </Table.Cell>
                    <Table.Cell className="font-medium text-gray-900 dark:text-white">
                      {faculty?.name}
                    </Table.Cell>
                    <Table.Cell>
                      <span
                        onClick={() => {
                          setShowModal(true);
                          setFacultyIdToDelete(faculty?._id);
                        }}
                        className="font-medium text-red-500 hover:underline cursor-pointer"
                      >
                        Delete
                      </span>
                    </Table.Cell>
                    {/* <Table.Cell>
                      <Link
                        className="text-teal-500 hover:underline"
                        to={`/update-course/${faculty?._id}`}
                      >
                        <span>Edit</span>
                      </Link>
                    </Table.Cell> */}
                  </Table.Row>
                </Table.Body>
              ))}
          </Table>
          {showMore && (
            <button
              onClick={handleShowMore}
              className="w-full text-teal-500 self-center text-sm py-7"
            >
              Show more
            </button>
          )}
        </>
      ) : (
        <p>You have no faculties yet!</p>
      )}
      <Modal
        show={showModal}
        onClose={() => setShowModal(false)}
        popup
        size="md"
      >
        <Modal.Header />
        <Modal.Body>
          <div className="text-center">
            <HiOutlineExclamationCircle className="h-14 w-14 text-red-600 dark:text-gray-200 mb-4 mx-auto" />
            <h3 className="mb-5 text-lg">
              Are you sure you want to delete this faculty?
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
                onClick={handleDeleteFaculty}
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
                onClick={() => setShowModal(false)}
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
